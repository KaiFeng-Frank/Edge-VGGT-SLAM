#!/usr/bin/env python3
"""Same original attention, CPU or DMA-imported Mali softmax, RK3588S only.

ABBAABBA over two persistent native MatMul contexts. No SDK GPU callback,
new weights, omitted keys, changed scale or CPU fallback. Compare precision
and the complete boundary; initialization and teardown are reported separately.
"""
import argparse
import ctypes as ct
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import sys
import time
import traceback

BASE=Path('/mnt/vggt-data/vggt-slam2'); E=BASE/'experiments'
PARENT_SHA='6302f2f2efa236b317ce5985923a21897945906b1d68cd21006c9422618c92d2'
LEGACY_SHA='f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2'
MALI_SHA='d57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda'
BRIDGE_SHA='73ff0c5ad33e0c334485553e9d8ab0764e6eb816a8f743b399531a58b619155e'
KERNEL_SHA='5cbb36e03e09c38db9e6769f067e3168e1cf7f67d719009601c4456e9f2b150a'
BASELINE_RAW='00545f97739c7b0a0748918cf766fcd808a9060e2edf6772ec0e88ed502e4fff'
N,PAD,H,D=1041,1056,16,64


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('/tmp/vggt-v119-shared-attention-r1'))
    p.add_argument('--source-root',type=Path,default=BASE/'src')
    args=p.parse_args()
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588S execution only')
    out=args.output_dir.resolve();require(str(out).startswith('/tmp/'),'Fresh private /tmp output required');out.mkdir(parents=True,exist_ok=False)
    faulthandler.enable(all_threads=True)
    env,affinity=dict(os.environ),set(os.sched_getaffinity(0));old_term=signal.getsignal(signal.SIGTERM)
    contexts=[];bridge=None;counted=None;pins={};began=time.perf_counter()
    report=dict(schema='v119_original_same_QKV_native_Mali_shared_attention_ABBA',status='running',script_sha256=sha(__file__),
        parent_source_sha256=PARENT_SHA,bridge_sha256=BRIDGE_SHA,kernel_sha256=KERNEL_SHA,
        original_model_only=True,queries=N,keys=N,heads=H,head_dim=D,padded_storage_keys=PAD,
        routes={'A':'native QK/PV with CPU FP32 softmax','B':'same QK/PV with DMA-imported Mali FP32 softmax'},
        all_original_keys_kept=True,padding_excluded_from_softmax=True,probability_storage='Half for both routes',
        source_QKV_Half_roundtrip_required=True,SDK_GPU_callback_used=False,CPU_fallback_enabled=False,
        CPU_FP32_model_or_ORT_executed=False,weights_or_checkpoint_deserialized=False,system_or_frequency_writes=False,
        complete_model_or_SLAM_or_realtime_validated=False,candidate_accepted=False,calls=[],cleanup_errors=[],
        init_seconds=None,destroy_seconds=None,
        timing_scope='All16 heads from owned FP32 QKV through conversions, dynamic K/V pack+upload, all32 MatMul calls, sync/bind, CPU or GPU softmax, GPU status/event handling, final owned output/finite checks. Init/destroy separate. CPU numerical validation, hashes and saving excluded.')
    def progress(stage):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-began,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temp=out/'result.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(out/'result.json')
        print(json.dumps(dict(stage=stage)),flush=True)
    def pin(path,digest):
        path=Path(path).resolve(strict=True);require(sha(path)==digest,'Pinned file differs: '+str(path));pins[path]=digest;return path
    def interrupted(number,frame):raise InterruptedError('External bounded controller interrupted shared attention')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        require({4,5,6,7}<=affinity,'A76 CPUs unavailable');os.sched_setaffinity(0,{4,5,6,7})
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',RKNN_LOG_LEVEL='0')
        for key in ('RKNN_DUMP_DIR','RKNN_DUMP_QUANT','RKNN_DUMP_TXT_TENSOR'):os.environ.pop(key,None)
        require(os.statvfs(out).f_bavail*os.statvfs(out).f_frsize>=128<<20,'Need128MiB free')
        pin(__file__,report['script_sha256'])
        parent=module(pin('/tmp/vggt-v118-source-r1/run_native_attention_board.py',PARENT_SHA),'_v119_frozen_native_product')
        pin(args.source_root/'third_party/vggt/vggt/layers/attention.py',parent.ATTENTION_SHA)
        abi_path=pin(args.source_root/'vggt_slam/rknn_linear.py',parent.ABI_SHA)
        sdk=pin('/usr/lib/librknnrt.so',parent.SDK_SHA)
        neon_path=pin(E/'vggt-v56-neon-build-r1/fp16_neon.so',parent.NEON_SHA)
        legacy=pin(args.source_root/'deploy/rk3588/validation/original_native_gpu/native_erf_gelu_opencl.py',LEGACY_SHA)
        mali=pin(E/'libmali-valhall-g610-g6p0-dummy.so',MALI_SHA)
        bp=pin(Path(__file__).parent/'native_attention_softmax_opencl.py',BRIDGE_SHA)
        kp=pin(Path(__file__).parent/'attention_softmax.cl',KERNEL_SHA)
        tp=pin(E/'vggt-v100-dino-taps-core1-r1/result.json',parent.TAP_SHA)
        cp=pin(E/'vggt-v100-qkv-counterfactual-r1/result.json',parent.CF_SHA)
        dp=pin('/tmp/vggt-v117-dump-r1/result.json',parent.DUMP_SHA)
        t,c,d=[json.loads(x.read_text()) for x in (tp,cp,dp)]
        require(t['final_baseline_byte_equal_all_calls'] and c['baseline_all_points_byte_equal_original_CPU']
            and c['QKV_substitution_calls']==1 and d['final_baseline_byte_equal'],'Frozen actual reference gates fail')
        import numpy as np
        def raw(v):return hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()
        def array(base,row):
            path=pin(base/row['file'],row['sha256']);v=np.load(path,allow_pickle=False)
            require(path.stat().st_size==row['bytes'] and list(v.shape)==row['shape'] and str(v.dtype)==row['dtype']
                and raw(v)==row['raw_sha256'] and np.isfinite(v).all(),'Reference tensor differs');return v
        row=t['calls'][0]['artifact'];zpath=pin(tp.parent/row['file'],row['sha256'])
        with np.load(zpath,allow_pickle=False) as z:qkv=np.ascontiguousarray(z['attn.qkv'])
        require(qkv.shape==(1,N,H*D*3) and qkv.dtype==np.float32 and raw(qkv)==parent.QKV_RAW
            and qkv.astype(np.float16).astype(np.float32).tobytes()==qkv.tobytes(),'Exact actual Half-representable QKV required')
        reference=array(cp.parent,c['artifacts']['attn.proj.input'])
        row=next(x for x in d['dump_files'] if x['file'].startswith('008_'))
        fused=np.ascontiguousarray(array(dp.parent/'dumps',row)[:,:,0,:].transpose(0,2,1))
        report.update(checkpoint=c['checkpoint'],QKV_raw_sha256=raw(qkv),CPU_reference_raw_sha256=raw(reference),fused_reference_raw_sha256=raw(fused))
        abi=module(abi_path,'_v119_typed_MatMul_ABI')
        require((ct.sizeof(abi.RKNNMatmulInfo),ct.sizeof(abi.RKNNMatmulIOAttr),ct.sizeof(abi.RKNNTensorMem))==(64,996,40),'ABI sizes differ')
        counted=parent.Counted(abi._API(str(sdk)));neon=ct.CDLL(str(neon_path))
        neon.f32_to_f16.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];neon.f32_to_f16.restype=None
        def half(v):
            v=np.ascontiguousarray(v,dtype=np.float32);h=np.empty(v.shape,np.float16)
            neon.f32_to_f16(v.ctypes.data,h.ctypes.data,v.size);require(np.isfinite(h).all(),'Half overflow');return h
        def compare(value,ref):
            v,r=value.astype(np.float64),ref.astype(np.float64);delta=v-r;bad=np.abs(delta)>1e-5+1e-4*np.abs(r)
            return dict(relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(r)),max_abs=float(np.abs(delta).max()),
                strict_rtol=1e-4,strict_atol=1e-5,strict_pass=not bool(bad.any()),outside_count=int(bad.sum()))
        def save(name,v):
            path=out/name
            with path.open('xb') as stream:np.save(stream,v,allow_pickle=False)
            return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),raw_sha256=raw(v),shape=list(v.shape),dtype=str(v.dtype))
        def platform_state():
            return {name:dict(frequency_hz=int((base/'cur_freq').read_text()),governor=(base/'governor').read_text().strip())
                for name,base in [('NPU',Path('/sys/class/devfreq/fdab0000.npu')),('GPU',Path('/sys/class/devfreq/fb000000.gpu'))]}
        report['platform_before']=platform_state()
        require(report['platform_before']['NPU']['frequency_hz']==1000000000,'External1GHz controller required')
        progress('initializing_native_contexts_and_shared_softmax')
        start=time.perf_counter()
        for k,n in ((D,PAD),(PAD,D)):
            product=parent.Product(abi,counted,np,k,n);contexts.append(product);product.open()
        qk,pv=contexts
        gpu=module(bp,'_v119_original_softmax_bridge')
        bridge=gpu.NativeAttentionSoftmaxBridge.__new__(gpu.NativeAttentionSoftmaxBridge)
        bridge.__init__(qk.memories['C'],pv.memories['A'],legacy_source_path=legacy,library_path=mali,kernel_source_path=kp)
        report['init_seconds']=time.perf_counter()-start
        report['bridge_initial']=bridge.metrics()
        def native_upload(product,name,value):
            if name=='A':packed=value.reshape(N,product.k//8,8).transpose(1,0,2)
            else:packed=value.reshape(product.k//32,32,product.n//16,16).transpose(2,0,3,1)
            np.copyto(product.views[name],packed,casting='no')
            product.sync(name,abi.RKNN_MEMORY_SYNC_TO_DEVICE);product.bind(name)
        def gpu_head(query,key,value):
            tick=time.perf_counter()
            native_upload(qk,'A',query);native_upload(qk,'B',key.T)
            now=time.perf_counter();upload_qk=now-tick
            abi._check(counted.rknn_matmul_run(qk.ctx),'QK run');end=time.perf_counter();qk_run=end-now
            shared=bridge.run(before_input=lambda:qk.sync('C',abi.RKNN_MEMORY_SYNC_FROM_DEVICE),
                before_output=lambda:pv.sync('A',abi.RKNN_MEMORY_SYNC_TO_DEVICE),
                after_output=lambda:pv.sync('A',abi.RKNN_MEMORY_SYNC_FROM_DEVICE))
            tick=time.perf_counter()
            pv.sync('A',abi.RKNN_MEMORY_SYNC_TO_DEVICE);pv.bind('A');native_upload(pv,'B',value)
            now=time.perf_counter();upload_pv=now-tick
            abi._check(counted.rknn_matmul_run(pv.ctx),'PV run');end=time.perf_counter();pv_run=end-now
            pv.sync('C',abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
            y=np.array(pv.views['C'].transpose(1,0,2),copy=True,order='C').reshape(N,D)
            require(np.isfinite(y).all(),'Nonfinite PV output')
            return y,dict(QK_pack_upload_bind_seconds=upload_qk,QK_SDK_run_seconds=qk_run,
                shared_softmax=shared,PV_upload_bind_seconds=upload_pv,PV_SDK_run_seconds=pv_run,
                PV_sync_owned_finite_seconds=time.perf_counter()-end)
        seen=set();validated_shared=False
        for iteration,route in enumerate('ABBAABBA'):
            require(platform_state()['NPU']['frequency_hz']==1000000000,'NPU controller frequency changed')
            progress('full_attention_'+str(iteration)+'_'+route)
            start=time.perf_counter();values=half(qkv).reshape(1,N,3,H,D).transpose(2,0,3,1,4);q,k,v=values[:,0]
            key=np.zeros((PAD,D),np.float16);value=np.zeros((PAD,D),np.float16)
            prob=np.zeros((N,PAD),np.float16) if route=='A' else None
            yheads=np.empty((H,N,D),np.float32);heads=[]
            for head in range(H):
                key[:N]=k[head];value[:N]=v[head]
                if route=='A':
                    scores,qt=qk.run(q[head],key.T);tick=time.perf_counter()
                    logits=np.ascontiguousarray(scores[:,:N]);logits*=np.float32(0.125);logits-=logits.max(axis=1,keepdims=True)
                    np.exp(logits,out=logits);logits/=logits.sum(axis=1,keepdims=True,dtype=np.float32);prob[:,:N]=half(logits)
                    require(not np.any(prob[:,N:]) and not np.any(value[N:]),'Padding must stay zero')
                    soft=time.perf_counter()-tick;yheads[head],pt=pv.run(prob,value)
                    heads.append(dict(head=head,QK=qt,softmax_crop_scale_probability_store_seconds=soft,PV=pt))
                else:
                    yheads[head],times=gpu_head(q[head],key,value);heads.append(dict(head=head,**times))
            y=np.ascontiguousarray(yheads.transpose(1,0,2).reshape(1,N,H*D));require(np.isfinite(y).all(),'Nonfinite attention')
            elapsed=time.perf_counter()-start
            if route=='A':require(raw(y)==BASELINE_RAW,'CPU-softmax route must reproduce original native output bytes')
            row=dict(index=iteration,route=route,first_call_for_route=route not in seen,complete_attention_seconds=elapsed,heads=heads,
                vs_original_CPU_same_QKV=compare(y,reference),vs_unchanged_fused_same_QKV=compare(y,fused),output=save('call'+str(iteration)+'.npy',y))
            report['calls'].append(row);seen.add(route)
            if route=='B' and not validated_shared:
                progress('same_scores_last_head_softmax_validation_outside_timing')
                qk.sync('C',abi.RKNN_MEMORY_SYNC_FROM_DEVICE);pv.sync('A',abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
                scores=np.array(qk.views['C'].transpose(1,0,2),copy=True,order='C').reshape(N,PAD)
                phalf=np.array(pv.views['A'].transpose(1,0,2),copy=True,order='C').reshape(N,PAD)
                require(np.isfinite(scores).all() and np.isfinite(phalf).all() and not np.any(phalf[:,N:]),'GPU probability finite/padding validation failed')
                logits=np.ascontiguousarray(scores[:,:N]);logits*=np.float32(0.125);logits-=logits.max(axis=1,keepdims=True)
                np.exp(logits,out=logits);logits/=logits.sum(axis=1,keepdims=True,dtype=np.float32)
                expected=half(logits)
                report['same_scores_validation']=dict(head=15,after_call=iteration,inside_benchmark_timing=False,
                    GPU_Half_vs_CPU_softmax_once_Half=compare(phalf[:,:N],expected),padding_all_zero=True,
                    scores=save('last-head-QK-native-product.npy',scores),GPU_probabilities=save('last-head-GPU-probabilities.npy',phalf),
                    CPU_probabilities_once_Half=save('last-head-CPU-probabilities.npy',expected))
                validated_shared=True
        report['subsequent_three_means_seconds']={route:statistics.mean(x['complete_attention_seconds'] for x in report['calls']
            if x['route']==route and not x['first_call_for_route']) for route in 'AB'}
        report['platform_after']=platform_state()
        report['status']='completed_shared_attention_component_not_full_model_accepted'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        tick=time.perf_counter();safe=bridge is None
        if bridge is not None:
            try:bridge.close()
            except BaseException as exc:report['cleanup_errors'].append('GPU close: '+str(exc))
            try:
                report['GPU_after_close']=bridge.metrics();g=report['GPU_after_close']
                safe=g['fully_released'] and g['live_events']==0 and g['queue_idle']
                report['cleanup_errors'].extend(g.get('cleanup_errors',[]))
                if report['status'].startswith('completed'):
                    if any(g[key]!=64 for key in ('calls_attempted','calls_completed','events_created','events_released')):
                        report['cleanup_errors'].append('Expected64 completed shared softmax calls and released events')
                    if g['objects_created']!=g['objects_released']:
                        report['cleanup_errors'].append('OpenCL object counts differ')
            except BaseException as exc:report['cleanup_errors'].append('GPU ownership unverified: '+str(exc))
        report['native_close_permitted_after_GPU_release']=safe
        resources=[]
        if safe:
            for context in reversed(contexts):
                try:
                    row=context.close();resources.append(row);report['cleanup_errors'].extend(row['cleanup_errors'])
                except BaseException as exc:report['cleanup_errors'].append('NPU close: '+str(exc))
        else:report['cleanup_errors'].append('NPU allocations retained because GPU may still use imported memory; process exit owns final reclamation')
        report.update(resources=resources,destroy_seconds=time.perf_counter()-tick)
        if report['init_seconds'] is not None and len(report['calls'])==8:
            report['summed_eight_call_lifecycle_seconds']=report['init_seconds']+report['destroy_seconds']+sum(x['complete_attention_seconds'] for x in report['calls'])
        if counted is not None:
            report.update(SDK_attempts=counted.calls,SDK_successes=counted.successes)
            s=counted.successes;balanced=safe and s.get('rknn_matmul_create',0)==s.get('rknn_matmul_destroy',0) and s.get('rknn_create_mem',0)==s.get('rknn_destroy_mem',0)
            report['resource_counts_balanced']=balanced and all(x['live_context']==x['live_memories']==0 for x in resources)
            if not report['resource_counts_balanced']:report['cleanup_errors'].append('Native resources unbalanced')
            if report['status'].startswith('completed'):
                expected=dict(rknn_matmul_run=256,rknn_matmul_create=2,rknn_matmul_destroy=2,
                    rknn_matmul_set_core_mask=2,rknn_create_mem=6,rknn_destroy_mem=6,rknn_matmul_set_io_mem=518,rknn_mem_sync=898)
                report['expected_SDK_successes']=expected
                if s!=expected or counted.calls!=expected:report['cleanup_errors'].append('SDK calls differ from complete ABBA schedule')
        for path,digest in pins.items():
            if not path.is_file() or sha(path)!=digest:report['cleanup_errors'].append('Pin changed: '+str(path))
        report['source_and_inputs_unchanged']=not report['cleanup_errors']
        try:
            os.sched_setaffinity(0,affinity);os.environ.clear();os.environ.update(env);signal.signal(signal.SIGTERM,old_term)
            report.update(affinity_restored=set(os.sched_getaffinity(0))==affinity,environment_restored=dict(os.environ)==env)
            require(report['affinity_restored'] and report['environment_restored'],'Process state restoration failed')
        except BaseException as exc:report['cleanup_errors'].append(str(exc))
        if report['cleanup_errors']:report['status']='failed_cleanup'
        progress('finished')
    return 0 if report['status'].startswith('completed') else 1


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
