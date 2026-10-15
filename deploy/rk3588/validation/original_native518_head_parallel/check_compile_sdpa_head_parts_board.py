#!/usr/bin/env python3
"""Board FP32 gate or noncalibrating FP16 build of unchanged original SDPA parts.

Check runs the three static graph shapes on genuine original Q/K/V, with all
four head ranges and original consumer reconstruction. Compile imports no ORT
and requires a pinned successful check and its saved outputs. No calibration,
weight substitution, pruning, alternate model or performance claim is made.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback
import types

EXPORTER_SHA='8ebf9183d8b04cf3769f5de0e873ef912551b88ce26dac56ab66db883329573c'
PARENT_COMPILER_SHA='ade8f932c267bdc2a9ff5854e305d87456984630025aa34c0c448499f5a6a81b'
CHECK_SCHEMA='v112_original_native518_SDPA_head_parts_FP32_check'
COMPILE_SCHEMA='v112_original_native518_SDPA_head_parts_FP16_compile'
CHECK_SUCCESS='checked_original_SDPA_parts_not_NPU_quality_accepted'
COMPILE_SUCCESS='compiled_original_SDPA_parts_not_quality_accepted'
FIXED_GATE=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5)
EXPECTED_CONFIG=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,
    mean_values=None,std_values=None,quant_img_RGB2BGR=False,model_pruning=False,
    remove_reshape=False,quantize_weight=False,enable_flash_attention=False)
BUILD=dict(do_quantization=False,dataset=None,auto_hybrid=False)
THREAD_ENV=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
            'CUDA_VISIBLE_DEVICES','TMPDIR')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def pin(path,digest,watched):
    path=Path(path).resolve(strict=True)
    require(len(digest)==64 and sha(path)==digest,'Pinned file differs: '+str(path))
    watched[str(path)]=digest;return path


def load_source(path,digest,name,watched):
    path=pin(path,digest,watched);m=types.ModuleType(name);m.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def compare(actual,reference):
    import numpy as np
    require(actual.dtype==reference.dtype==np.float32 and actual.shape==reference.shape and
        np.isfinite(actual).all() and np.isfinite(reference).all(),'Complete finite FP32 comparison required')
    diff=actual.astype(np.float64)-reference.astype(np.float64)
    norm=float(np.linalg.norm(reference.astype(np.float64).ravel()))
    l2=float(np.linalg.norm(diff.ravel()))
    relative=l2/norm if norm else (0. if l2==0. else 1e300)
    maxabs=float(np.max(np.abs(diff)))
    strict=bool(np.allclose(actual,reference,rtol=FIXED_GATE['rtol'],atol=FIXED_GATE['atol']))
    return dict(relative_l2=relative,max_abs=maxabs,strict_allclose=strict,
        byte_equal=actual.tobytes()==reference.tobytes(),
        fixed_gate_passed=relative<=1e-5 and maxabs<=1e-3 and strict)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('check','compile'),required=True)
    p.add_argument('--export',type=Path,required=True);p.add_argument('--export-sha256',required=True)
    p.add_argument('--exporter-source',type=Path,required=True);p.add_argument('--exporter-sha256',required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--capture-sha256',required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--check-receipt',type=Path);p.add_argument('--check-sha256')
    p.add_argument('--parent-compiler-source',type=Path);p.add_argument('--parent-compiler-sha256')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cpu-affinity',type=int,nargs='+',default=[4,6])
    a=p.parse_args()
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
        'All graph inference/build execution in this driver is restricted to RK3588/S')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(schema=CHECK_SCHEMA if a.mode=='check' else COMPILE_SCHEMA,mode=a.mode,status='failed',
        script_sha256=sha(__file__),original_model_only=True,component_only=True,
        candidate_accepted=False,full_model_or_trajectory_or_realtime_accepted=False,
        fixed_gate=FIXED_GATE,SDK_or_NPU_execution=a.mode=='compile',GPU_execution=False,
        inference_execution=a.mode=='check',system_or_frequency_writes=False,cleanup_errors=[])
    watched={str(Path(__file__).resolve()):r['script_sha256']}
    oldaff=os.sched_getaffinity(0);oldenv={k:os.environ.get(k) for k in THREAD_ENV};oldcwd=os.getcwd()
    handlers={};sdk=None;began=time.perf_counter()
    def interrupted(number,frame):raise InterruptedError('Termination signal '+str(number))
    try:
        require(a.exporter_sha256==EXPORTER_SHA,'Frozen extractor source required')
        E=load_source(a.exporter_source,EXPORTER_SHA,'v112_frozen_exporter',watched)
        ep=pin(a.export,a.export_sha256,watched);e=json.loads(ep.read_text())
        require(e['schema']==E.SCHEMA and e['status']==E.SUCCESS and e['script_sha256']==EXPORTER_SHA and
            e['source_and_inputs_unchanged'] and e['cleanup_errors']==[] and
            e['proof']['original_attention_node_protobufs_unchanged'] and
            e['proof']['all18_parent_learned_parameters_match_original_capture'] and
            e['parts']==E.PARTS and a.capture_sha256==E.CAPTURE_SHA,'Exact clean original extraction required')
        cp,c=E.capture_receipt(a.capture);watched[str(cp)]=E.CAPTURE_SHA
        require(e['capture']['sha256']==E.CAPTURE_SHA and e['checkpoint']==E.CHECKPOINT,'Capture/checkpoint lineage differs')
        for name,digest in c['source_hashes'].items():
            if name.startswith('third_party/vggt/'):pin(a.source_root/name,digest,watched)
        arrays,ap=E.load_capture_arrays(cp,c);watched.update(ap)
        graph_rows={row['key']:row for row in e['models']}
        require(set(graph_rows)=={'heads16','heads6','heads5'},'Exact three graph shapes required')
        graph_paths={}
        for key,row in graph_rows.items():
            require(Path(row['file']).name==row['file'] and row['heads']==int(key[5:]),'Direct graph/shape identity differs')
            file=pin(ep.parent/row['file'],row['sha256'],watched)
            require(file.stat().st_size==row['bytes'],'Graph size differs');graph_paths[key]=file
        os.sched_setaffinity(0,set(a.cpu_affinity));require(set(a.cpu_affinity).issubset({4,5,6,7}) and a.cpu_affinity,'A76-local affinity required')
        for name in THREAD_ENV:os.environ[name]='' if name=='CUDA_VISIBLE_DEVICES' else ('1' if name!='TMPDIR' else str(out))
        os.chdir(out)
        for number in (signal.SIGTERM,signal.SIGINT):handlers[number]=signal.signal(number,interrupted)
        r.update(export_receipt=dict(file=str(ep),sha256=a.export_sha256),capture=dict(file=str(cp),sha256=E.CAPTURE_SHA),
            checkpoint=E.CHECKPOINT,parts=copy.deepcopy(E.PARTS),original_SDPA_arguments=c['SDPA_arguments'],
            original_source_hashes=c['source_hashes'],exporter_source_sha256=EXPORTER_SHA)
        if a.mode=='check':
            import numpy as np
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            q,k,v=[arrays[name] for name in ('q_post_rope','k_post_rope','v')]
            cases=[dict(id='A16',model_key='heads16',start=0,end=16)]+[
                dict(id='part'+str(part['index']),model_key=part['model_key'],start=part['start'],end=part['end']) for part in E.PARTS]
            checks=[];results={}
            for case in cases:
                session=ort.InferenceSession(str(graph_paths[case['model_key']]),sess_options=options,providers=['CPUExecutionProvider'])
                require([v.name for v in session.get_inputs()]==list(E.INPUT_NAMES) and
                    [v.name for v in session.get_outputs()]==['attention_values'],'Original attention IO order differs')
                begin,end=case['start'],case['end']
                inputs=[np.ascontiguousarray(ar[:,begin:end]) for ar in (q,k,v)]
                output=session.run(['attention_values'],dict(zip(E.INPUT_NAMES,inputs)))[0]
                output=np.array(output,copy=True,order='C');del session
                result=compare(output,np.ascontiguousarray(arrays['attention_tail'][:,begin:end]))
                path=out/(case['id']+'.npy');np.save(path,output,allow_pickle=False)
                checks.append(dict(**case,comparison=result,actual_output=dict(file=path.name,bytes=path.stat().st_size,
                    sha256=sha(path),shape=list(output.shape),dtype='float32',raw_sha256=hashlib.sha256(output.tobytes()).hexdigest())))
                results[case['id']]=output
                require(result['fixed_gate_passed'],'Original full FP32 gate failed: '+case['id'])
            joined=np.ascontiguousarray(np.concatenate([results['part'+str(i)] for i in range(3)],axis=1))
            proj=np.ascontiguousarray(joined.transpose(0,2,1,3).reshape(1,E.N,1024))
            extra=dict(joined_vs_original=compare(joined,arrays['attention_tail']),joined_vs_A16=compare(joined,results['A16']),
                restored_proj_input_vs_original=compare(proj,arrays['proj_input']))
            require(all(item['fixed_gate_passed'] for item in extra.values()),'Joined original complete head/consumer FP32 gate failed')
            r.update(status=CHECK_SUCCESS,checks=checks,reassembly_checks=extra,original_FP32_gate_passed=True,
                models=e['models'],versions=dict(onnxruntime=ort.__version__,numpy=np.__version__),sdk_released_not_applicable=True)
        else:
            require(a.check_receipt is not None and a.check_sha256 and a.parent_compiler_source is not None and
                a.parent_compiler_sha256==PARENT_COMPILER_SHA,'Pinned board FP32 gate and frozen build config required')
            chp=pin(a.check_receipt,a.check_sha256,watched);ch=json.loads(chp.read_text())
            require(ch['schema']==CHECK_SCHEMA and ch['mode']=='check' and ch['status']==CHECK_SUCCESS and
                ch['script_sha256']==r['script_sha256'] and ch['export_receipt']['sha256']==a.export_sha256 and
                ch['capture']['sha256']==E.CAPTURE_SHA and ch['fixed_gate']==FIXED_GATE and
                ch['original_FP32_gate_passed'] and ch['source_and_inputs_unchanged'] and not ch['cleanup_errors'] and
                ch['affinity_restored'] and ch['environment_restored'] and ch['cwd_restored'] and ch['models']==e['models'],
                'Clean successful same-source FP32 gate required before build')
            import numpy as np
            require([x['id'] for x in ch['checks']]==['A16','part0','part1','part2'],'Four complete head-range checks required')
            saved={}
            for item in ch['checks']:
                meta=item['actual_output'];require(Path(meta['file']).name==meta['file'],'Direct saved FP32 output required')
                path=pin(chp.parent/meta['file'],meta['sha256'],watched);actual=np.load(path,allow_pickle=False)
                require(path.stat().st_size==meta['bytes'] and list(actual.shape)==meta['shape'] and
                    str(actual.dtype)==meta['dtype']=='float32' and hashlib.sha256(actual.tobytes()).hexdigest()==meta['raw_sha256'],
                    'Saved board check output identity differs')
                ref=np.ascontiguousarray(arrays['attention_tail'][:,item['start']:item['end']])
                require(compare(actual,ref)==item['comparison'] and item['comparison']['fixed_gate_passed'],
                    'Saved original FP32 check does not independently recompute')
                saved[item['id']]=actual
            joined=np.ascontiguousarray(np.concatenate([saved['part'+str(i)] for i in range(3)],axis=1))
            reassembly=dict(joined_vs_original=compare(joined,arrays['attention_tail']),joined_vs_A16=compare(joined,saved['A16']),
                restored_proj_input_vs_original=compare(np.ascontiguousarray(joined.transpose(0,2,1,3).reshape(1,E.N,1024)),arrays['proj_input']))
            require(reassembly==ch['reassembly_checks'] and all(x['fixed_gate_passed'] for x in reassembly.values()),'Joined FP32 gate differs')
            H=load_source(a.parent_compiler_source,PARENT_COMPILER_SHA,'v112_frozen_build_configuration',watched)
            require(H.CONFIG==EXPECTED_CONFIG,'Original nonquantizing compiler config changed')
            r.update(check_receipt=dict(file=str(chp),sha256=a.check_sha256),checks=ch['checks'],reassembly_checks=reassembly,
                original_FP32_gate_passed=True,configuration=copy.deepcopy(H.CONFIG),build_options=BUILD,
                parent_compiler_source_sha256=PARENT_COMPILER_SHA,models=[],calls=[])
            require(importlib.metadata.version('rknn-toolkit2').split('+')[0].startswith('2.3.2'),'RKNN toolkit2.3.2 required')
            from rknn.api import RKNN
            for key,graph in graph_paths.items():
                sdk=RKNN(verbose=False,verbose_file=str(out/(key+'-sdk.log')))
                target=out/(key+'.rknn')
                try:
                    for operation,fn in [('config',lambda:sdk.config(**H.CONFIG)),('load_onnx',lambda:sdk.load_onnx(model=str(graph))),
                        ('build',lambda:sdk.build(**BUILD)),('export_rknn',lambda:sdk.export_rknn(str(target)))]:
                        tick=time.perf_counter();code=fn();r['calls'].append(dict(model_key=key,operation=operation,returncode=code,seconds=time.perf_counter()-tick))
                        atomic(out/'result.json',r);require(code in (None,0),key+' '+operation+' failed')
                    row=copy.deepcopy(graph_rows[key]);row['rknn']=dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target));r['models'].append(row)
                finally:
                    code=sdk.release();r.setdefault('release_codes',[]).append(dict(model_key=key,returncode=code))
                    require(code in (None,0),'SDK release failed');sdk=None
            r.update(status=COMPILE_SUCCESS,sdk_released=True,versions={name:importlib.metadata.version(name) for name in ('rknn-toolkit2','numpy','onnx')})
    except BaseException as error:r.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if sdk is not None:
            try:
                code=sdk.release();require(code in (None,0),'Retry SDK release failed')
            except BaseException as error:r['cleanup_errors'].append('SDK release: '+repr(error))
        try:
            os.chdir(oldcwd);r['cwd_restored']=os.getcwd()==oldcwd
            os.sched_setaffinity(0,oldaff);r['affinity_restored']=os.sched_getaffinity(0)==oldaff
            for name,value in oldenv.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value
            r['environment_restored']=all(os.environ.get(name)==value for name,value in oldenv.items())
            for number,handler in handlers.items():signal.signal(number,handler)
        except BaseException as error:r['cleanup_errors'].append('Process restoration: '+repr(error))
        r['source_and_inputs_unchanged']=all(sha(path)==digest for path,digest in watched.items())
        if not r['source_and_inputs_unchanged']:r['cleanup_errors'].append('Pinned source/input changed')
        if r['cleanup_errors']:r['status']='failed_cleanup'
        r['elapsed_seconds']=time.perf_counter()-began
        atomic(out/'result.json',r)
    print(json.dumps({k:r.get(k) for k in ('mode','status','models','error')}),flush=True)
    return 0 if r['status'] in (CHECK_SUCCESS,COMPILE_SUCCESS) else 1


if __name__=='__main__':raise SystemExit(main())
