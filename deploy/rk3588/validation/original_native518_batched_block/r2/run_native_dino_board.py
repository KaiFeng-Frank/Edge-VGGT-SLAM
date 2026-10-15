#!/usr/bin/env python3
"""Board-only original full DINO0 ABBA: fused RKNN versus batched native attention IO.

The candidate preserves original weights, layers, tokens and formulas. Explicit
Half linear/attention boundaries are tested numerical changes, not equivalence.
This complete block is a screen; full-model SLAM acceptance is separate.
"""
import argparse
import copy
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

B=Path('/mnt/vggt-data/vggt-slam2'); E=B/'experiments'; SRC=B/'src'
PARENT_SHA='6302f2f2efa236b317ce5985923a21897945906b1d68cd21006c9422618c92d2'
OWNED_SHA='0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
CAPTURE_SHA='17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
WEIGHT_CAPTURE_SHA='01931ff710097ce06782de9f99431a20cb5cfdce472ded9e45f398df9c8bc8bb'
WEIGHT_SHA='034497947f816fab655c11109d479a7fd94b582f7dd8e81c17977f9b9c593ae2'
BASE_MODEL_SHA='fe8acac49eb10b1215edf1d4c348b69da3ddfed53f16c2e202b0e3f6428b32c0'
BASE_OUTPUT_RAW='e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554'
LEGACY_SHA='f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2'
MALI_SHA='d57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda'
R121_RESULT_SHA='baa246379666a5b2320d62b7935ae553d398feef8be21ef58e27d727f5a295ff'
R121_B_RAW='70de59362e3bb03eb875bc3ee9be56a7af538b4098e662d9fd6b817d5c3c3e0d'
DINO_KERNEL_SHA='da7e8d3c15bd4f67732b6d4a727f065eef0349635ad7b81770b08ecf9719dc47'
SOFTMAX_SHA='5cbb36e03e09c38db9e6769f067e3168e1cf7f67d719009601c4456e9f2b150a'
CHECKPOINT=dict(bytes=5026874952,sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
SHAPE=(1,1041,1024)


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''): h.update(part)
    return h.hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--source-manifest-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588S only')
    out=a.output_dir.resolve(); require(str(out).startswith('/tmp/'),'Fresh private /tmp output required')
    out.mkdir(parents=True,exist_ok=False); faulthandler.enable(all_threads=True)
    started=time.perf_counter(); env=dict(os.environ); affinity=set(os.sched_getaffinity(0)); oldterm=signal.getsignal(signal.SIGTERM)
    pins={}; original=None; candidate=None; counted=None
    r=dict(schema='v122_original_native518_complete_DINO0_batched_native_ABBA',status='running',script_sha256=sha(__file__),
        checkpoint=CHECKPOINT,original_model_only=True,logical_shape=list(SHAPE),schedule='ABBAABBA',
        all_original_layers_tokens_weights_and_formulas_retained=True,reference_intermediate_substituted=False,
        static_weight_storage='original FP32 weights converted once to FP16 by pinned board NEON; no quantization or retraining',
        activations='GPU FP32 LN/bias/residual/Erf and stable softmax; native NPU products FP16 inputs/FP32 output; explicit Half boundaries',
        checkpoint_deserialized=False,NN_or_SDK_execution_on_host=False,system_or_frequency_writes=False,
        candidate_accepted=False,complete_model_SLAM_trajectory_or_realtime_validated=False,
        weighted_and_native_core_mask=1,calls=[],cleanup_errors=[],initialization_seconds={},
        timing_scope='Each complete block includes input upload/validation, all four static-weight products and32 attention products, all GPU operations and synchronizations, final owned output and finite check. Init/destroy separate. Comparisons, file IO, and first-B buffer capture outside. No intermediate reference fed into either route.')
    def progress(stage):
        r.update(stage=stage,elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temp=out/'result.tmp'; temp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n'); temp.replace(out/'result.json')
        print(json.dumps(dict(stage=stage)),flush=True)
    def pin(path,digest,size=None):
        path=Path(path).resolve(strict=True)
        require((size is None or path.stat().st_size==size) and sha(path)==digest,'Pin differs: '+str(path))
        pins[path]=digest; return path
    def state():
        return {key:dict(frequency_hz=int((base/'cur_freq').read_text()),governor=(base/'governor').read_text().strip())
            for key,base in [('NPU',Path('/sys/class/devfreq/fdab0000.npu')),('GPU',Path('/sys/class/devfreq/fb000000.gpu'))]}
    def term(number,frame): raise InterruptedError('External180s/75C controller stopped experiment')
    signal.signal(signal.SIGTERM,term)
    try:
        require({4,5,6,7}<=affinity,'A76 unavailable'); os.sched_setaffinity(0,{4,5,6,7})
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',RKNN_LOG_LEVEL='0')
        for key in ('RKNN_DUMP_DIR','RKNN_DUMP_QUANT','RKNN_DUMP_TXT_TENSOR'): os.environ.pop(key,None)
        require(os.statvfs(out).f_bavail*os.statvfs(out).f_frsize>=160<<20,'Need160MiB free')
        sp=pin(a.source_manifest,a.source_manifest_sha256); bundle=json.loads(sp.read_text())
        names={'run_native_dino_board.py','native_dino_runtime.py','native_dino_bridge_opencl.py','dino0_native_bridge.cl','attention_softmax.cl','dino0_batched_bridge.cl'}
        require(set(bundle)==names,'Exact six-source-only bundle required')
        sources={name:pin(sp.parent/name,row['sha256'],row['bytes']) for name,row in bundle.items()}
        require(sources['run_native_dino_board.py']==Path(__file__).resolve(),'Driver differs from pinned bundle')
        require(bundle['attention_softmax.cl']['sha256']==SOFTMAX_SHA,'Frozen original full1041 softmax required')
        require(bundle['dino0_native_bridge.cl']['sha256']==DINO_KERNEL_SHA,'Frozen original full1041 scalar kernels required')
        r['source_bundle']=bundle; r['source_manifest_sha256']=a.source_manifest_sha256
        parent=module(pin('/tmp/vggt-v118-source-r1/run_native_attention_board.py',PARENT_SHA),'_v122_native_product')
        abi=module(pin(SRC/'vggt_slam/rknn_linear.py',parent.ABI_SHA),'_v122_native_abi')
        sdk=pin('/usr/lib/librknnrt.so',parent.SDK_SHA)
        pin(SRC/'third_party/vggt/vggt/layers/attention.py',parent.ATTENTION_SHA)
        require((ct.sizeof(abi.RKNNMatmulInfo),ct.sizeof(abi.RKNNMatmulIOAttr),ct.sizeof(abi.RKNNTensorMem))==(64,996,40),'ABI differs')
        neon=ct.CDLL(str(pin(E/'vggt-v56-neon-build-r1/fp16_neon.so',parent.NEON_SHA)))
        legacy=pin(SRC/'deploy/rk3588/validation/original_native_gpu/native_erf_gelu_opencl.py',LEGACY_SHA)
        mali=pin(E/'libmali-valhall-g610-g6p0-dummy.so',MALI_SHA)
        wc=pin(E/'vggt-v99-dino-reference-r1/capture.json',WEIGHT_CAPTURE_SHA); wref=json.loads(wc.read_text())
        cp=pin(E/'vggt-v100-dino0-intermediates-r1/capture.json',CAPTURE_SHA); capture=json.loads(cp.read_text())
        require(wref['checkpoint']==capture['checkpoint']==CHECKPOINT and wref['parameter_count']==14
            and wref['model_state_unchanged'] and wref['source_and_inputs_unchanged'] and wref['hooks_removed']
            and capture['final_comparison']['byte_equal_v99'] and capture['final_comparison']['raw_sha256_equal_v99'],'Original parameter/reference provenance fails')
        arch=wref['architecture']
        require(arch['heads']==16 and arch['embed_dim']==1024 and arch['epsilons']==dict(norm1=1e-6,norm2=1e-6)
            and not arch['qk_norm'] and arch['rope'] is None and arch['gelu_approximate']=='none','Original block architecture differs')
        require(wref['parameters']==capture['original_parameters'],'Original capture parameter identities differ')
        import numpy as np
        def raw(v): return hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()
        wp=pin(wc.parent/wref['weights']['file'],WEIGHT_SHA,50396702)
        with np.load(wp,allow_pickle=False) as z:
            require(set(z.files)==set(wref['parameters']),'Exactly original14 parameter keys required')
            params={key:np.ascontiguousarray(z[key]) for key in z.files}
        for key,value in params.items():
            row=wref['parameters'][key]
            require(value.dtype==np.float32 and list(value.shape)==row['shape'] and raw(value)==row['raw_sha256']
                and np.isfinite(value).all(),'Original learned parameter differs: '+key)
        r['original_parameters']=wref['parameters']; r['weights']=wref['weights']; r['architecture']=arch
        previous=pin('/tmp/vggt-v121-block-r1/result.json',R121_RESULT_SHA)
        prev=json.loads(previous.read_text())
        require(prev['status']=='completed_original_DINO0_ABBA_not_full_model_accepted' and not prev['cleanup_errors']
            and prev['source_and_inputs_unchanged'] and prev['original_parameters']==wref['parameters']
            and all(c['output']['raw_sha256']==R121_B_RAW for c in prev['calls'] if c['route']=='B'), 'Verified previous complete original native result required')
        r['previous_serial_receipt_sha256']=R121_RESULT_SHA
        r['previous_serial_output_raw_sha256']=R121_B_RAW
        r['only_scheduling_changed']=True
        def load(row):
            value=np.load(pin(cp.parent/row['file'],row['sha256'],row['bytes']),allow_pickle=False)
            require(value.dtype==np.float32 and list(value.shape)==row['shape'] and raw(value)==row['raw_sha256']
                and np.isfinite(value).all(),'Captured original tensor differs'); return value
        def save(name,value):
            path=out/name
            with path.open('xb') as stream: np.save(stream,value,allow_pickle=False)
            return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),raw_sha256=raw(value),shape=list(value.shape),dtype=str(value.dtype))
        def compare(y,z):
            yd,zd=y.astype(np.float64),z.astype(np.float64); d=yd-zd; bad=np.abs(d)>1e-5+1e-4*np.abs(zd)
            l2=float(np.linalg.norm(d)/np.linalg.norm(zd)); maximum=float(np.abs(d).max())
            return dict(relative_l2=l2,max_abs=maximum,strict_rtol=1e-4,strict_atol=1e-5,strict_pass=not bool(bad.any()),
                outside_count=int(bad.sum()),fixed_reference_gate_passed=l2<=1e-5 and maximum<=1e-3,byte_equal=y.tobytes()==z.tobytes())
        x=load(capture['artifacts']['input']); reference=load(capture['artifacts']['output']); input_hash=raw(x)
        require(x.shape==reference.shape==SHAPE,'Complete1041 input required')
        r['original_input']=capture['artifacts']['input']; r['original_output_reference']=capture['artifacts']['output']
        model=pin(E/'vggt-v99-dino-compile-r1/native-block.rknn',BASE_MODEL_SHA,26517686)
        O=module(pin(E/'vggt-v100-source-r1/native_heads_runtime.py',OWNED_SHA),'_v122_owned_baseline')
        pin(E/'vggt-v99-source-r1/run_native_block_r1.py',O.HELPER_SHA)
        pin(SRC/'deploy/rk3588/validation/run-v12-rknn-block-board.py',O.ABI_SHA)
        gpu=module(sources['native_dino_bridge_opencl.py'],'_v122_native_gpu')
        runtime=module(sources['native_dino_runtime.py'],'_v122_native_runtime')
        counted=parent.Counted(abi._API(str(sdk)))
        r['platform_before']=state(); require(r['platform_before']['NPU']['frequency_hz']==1000000000,'External1GHz controller required')
        progress('initializing_original_fused_baseline'); tick=time.perf_counter()
        original=O.OwnedContext.__new__(O.OwnedContext)
        original.__init__(model,BASE_MODEL_SHA,input_names=['input'],input_shapes=[SHAPE],output_names=['output'],output_shapes=[SHAPE],core_mask=1,flags=0)
        r['initialization_seconds']['baseline']=time.perf_counter()-tick
        progress('initializing_complete_native_block'); tick=time.perf_counter()
        candidate=runtime.NativeDino.__new__(runtime.NativeDino)
        candidate.__init__(parent,gpu,abi,counted,np,neon,params,legacy=legacy,mali=mali,
            kernel=sources['dino0_native_bridge.cl'],softmax=sources['attention_softmax.cl'],batch_kernel=sources['dino0_batched_bridge.cl'])
        r['initialization_seconds']['candidate']=time.perf_counter()-tick
        r['GPU_initial']=copy.deepcopy(candidate.initial_gpu_metrics)
        seen=set()
        for index,route in enumerate('ABBAABBA'):
            require(state()['NPU']['frequency_hz']==1000000000,'Frequency controller changed'); progress('complete_block_'+str(index)+'_'+route)
            tick=time.perf_counter()
            if route=='A':
                y=original.run([x])[0]; stages=copy.deepcopy(original.last_timings)
            else:
                y=candidate.run(x); stages=copy.deepcopy(candidate.last_timings)
            require(y.shape==SHAPE and y.dtype==np.float32 and np.isfinite(y).all(),'Finite complete original block output required')
            elapsed=time.perf_counter()-tick
            if route=='A': require(raw(y)==BASE_OUTPUT_RAW,'Original fused baseline output changed')
            row=dict(index=index,route=route,first_for_route=route not in seen,complete_block_seconds=elapsed,stages=stages,
                vs_original_CPU=compare(y,reference),residual_update_vs_original_CPU=compare(y-x,reference-x),output=save('call'+str(index)+'.npy',y))
            if route=='B' and route not in seen:
                row['diagnostics_outside_timing']={name:save('native-'+name+'.npy',value) for name,value in candidate.diagnostics().items()}
            if route=='B': row['byte_equal_previous_serial_native_output']=raw(y)==R121_B_RAW
            r['calls'].append(row); seen.add(route)
            if route=='B': require(row['byte_equal_previous_serial_native_output'],'Batched schedule changed original native output bytes')
        require(raw(x)==input_hash,'Original input modified'); r['original_input_unchanged']=True
        r['subsequent_three_means_seconds']={route:statistics.mean(c['complete_block_seconds'] for c in r['calls'] if c['route']==route and not c['first_for_route']) for route in 'AB'}
        r['platform_after']=state(); r['status']='completed_original_DINO0_ABBA_not_full_model_accepted'
    except BaseException as error:
        r.update(status='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        tick=time.perf_counter()
        if candidate is not None:
            try: candidate.close()
            except BaseException as error: r['cleanup_errors'].append('Candidate close: '+str(error))
            r['candidate_cleanup']=copy.deepcopy(getattr(candidate,'cleanup',{}))
            r['GPU_after_close']=copy.deepcopy(getattr(candidate,'final_gpu_metrics',{}))
        if original is not None:
            try:
                original.close(); m=original.metrics(); r['baseline_resources']=m
                require(m['resource_counts_balanced'] and not m['live_contexts'] and not m['live_output_buffers'] and not m['cleanup_errors'],'Baseline ownership unbalanced')
            except BaseException as error: r['cleanup_errors'].append('Baseline close: '+str(error))
        r['destroy_seconds']=time.perf_counter()-tick
        if counted is not None: r.update(MatMul_attempts=counted.calls,MatMul_successes=counted.successes)
        if r['status'].startswith('completed'):
            expected=dict(rknn_matmul_create=36,rknn_matmul_set_core_mask=36,rknn_create_mem=108,rknn_matmul_set_io_mem=384,
                rknn_mem_sync=972,rknn_matmul_run=144,rknn_destroy_mem=108,rknn_matmul_destroy=36)
            r['expected_MatMul_successes']=expected
            if counted.calls!=expected or counted.successes!=expected: r['cleanup_errors'].append('Native call ledger differs')
            g=r['GPU_after_close']
            if not g.get('fully_released') or g.get('events_created')!=28 or g.get('events_released')!=28 or g.get('objects_created')!=g.get('objects_released'):
                r['cleanup_errors'].append('GPU kernel or release ledger differs')
            r['summed_eight_call_lifecycle_seconds']=sum(r['initialization_seconds'].values())+r['destroy_seconds']+sum(c['complete_block_seconds'] for c in r['calls'])
        changed=[str(path) for path,digest in pins.items() if not path.is_file() or sha(path)!=digest]
        r['source_and_inputs_unchanged']=not changed; r['cleanup_errors'].extend('Changed pin: '+path for path in changed)
        try:
            os.sched_setaffinity(0,affinity); os.environ.clear(); os.environ.update(env); signal.signal(signal.SIGTERM,oldterm)
            r.update(affinity_restored=set(os.sched_getaffinity(0))==affinity,environment_restored=dict(os.environ)==env)
            require(r['affinity_restored'] and r['environment_restored'],'Process state restoration failed')
        except BaseException as error: r['cleanup_errors'].append(str(error))
        if r['cleanup_errors']: r['status']='failed_cleanup'
        progress('finished')
    return 0 if r['status'].startswith('completed') else 1


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
