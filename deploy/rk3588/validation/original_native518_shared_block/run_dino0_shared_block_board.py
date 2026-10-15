#!/usr/bin/env python3
"""Persistent complete original DINO0: fused versus true cuts/shared attention.

The original input feeds both the actual prefix and suffix residual. No saved
QKV or attention is substituted. Each complete call includes all SDK I/O,
original attention, output copies and finite checks; initialization is separate.
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
OWNED_SHA='0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
R119_SHA='10b15ffa734866283aede7ac865ab698114d204dd4f691d42cb0e0450059b092'
RUNTIME_SHA='82ec9d4ad838c84dcd02309a9da7d73db68b77d30525d1308f3e482907844968'
EXPORTER_SHA='3b61eddfd7cae104d0ad716b2b06b24d234dcea8f3ef94be0230e528d277752c'
CHECKER_SHA='7b673fdb2c4336773611d02be25a9eccb884310fbcda0beca759d46511a67312'
CAPTURE_SHA='17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
BASE_COMPILE_SHA='4864b5ac075e75d5c0e2600903792a510d88851c1c8040e22135440d03ec2b4e'
BASE_MODEL_SHA='fe8acac49eb10b1215edf1d4c348b69da3ddfed53f16c2e202b0e3f6428b32c0'
BASE_OUTPUT_RAW='e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554'
PARENT_ONNX_SHA='d8ce26c2f9cb3018b9a24da9f722937569067a5cffde650288f0073743e547e8'
CHECKPOINT=dict(bytes=5026874952,sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
SHAPE=(1,1041,1024); QKV=(1,1041,3072)
CONFIG=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,mean_values=None,std_values=None,
    quant_img_RGB2BGR=False,model_pruning=False,remove_reshape=False,quantize_weight=False,enable_flash_attention=False)
BUILD=dict(do_quantization=False,dataset=None,auto_hybrid=False)


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('export','check','compile'):
        p.add_argument('--'+key,type=Path,required=True);p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588S only')
    out=a.output_dir.resolve();require(str(out).startswith('/tmp/'),'Private fresh /tmp output required');out.mkdir(parents=True,exist_ok=False)
    faulthandler.enable(all_threads=True);started=time.perf_counter()
    env,affinity=dict(os.environ),set(os.sched_getaffinity(0));old_term=signal.getsignal(signal.SIGTERM)
    pins={};weighted=[];shared=None;counted=None
    r=dict(schema='v120_original_native518_complete_DINO0_shared_attention_ABBA',status='running',script_sha256=sha(__file__),
        original_model_only=True,checkpoint=CHECKPOINT,logical_shape=list(SHAPE),schedule='ABBAABBA',all31_original_nodes_accounted=False,
        original_layers_tokens_and_weights_retained=True,reference_QKV_or_attention_substituted=False,original_input_used_for_suffix_residual=True,
        NN_or_SDK_execution_on_host=False,checkpoint_deserialized=False,system_or_frequency_writes=False,candidate_accepted=False,
        complete_model_SLAM_trajectory_or_realtime_validated=False,weighted_core_mask=1,MatMul_core_mask=1,calls=[],cleanup_errors=[],
        timing_scope='Each persistent-context complete block call: input check, SDK input conversion/upload/run/get/release and owned output. B includes actual prefix→all16 full1041 shared attention→actual original suffix. Original residual x preserved. Context initialization and teardown separate; numerical checks/hash/saving outside.',
        initialization_seconds={},destroy_seconds=None)
    def progress(stage):
        r.update(stage=stage,elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temp=out/'result.tmp';temp.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');temp.replace(out/'result.json');print(json.dumps(dict(stage=stage)),flush=True)
    def pin(path,digest,size=None):
        path=Path(path).resolve(strict=True);require((size is None or path.stat().st_size==size) and sha(path)==digest,'Pinned artifact differs: '+str(path));pins[path]=digest;return path
    def platform_state():
        return {name:dict(frequency_hz=int((base/'cur_freq').read_text()),governor=(base/'governor').read_text().strip())
            for name,base in [('NPU',Path('/sys/class/devfreq/fdab0000.npu')),('GPU',Path('/sys/class/devfreq/fb000000.gpu'))]}
    def interrupted(number,frame):raise InterruptedError('External bounded controller interrupted complete DINO0')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        require({4,5,6,7}<=affinity,'A76 CPUs unavailable');os.sched_setaffinity(0,{4,5,6,7})
        os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',RKNN_LOG_LEVEL='0')
        for key in ('RKNN_DUMP_DIR','RKNN_DUMP_QUANT','RKNN_DUMP_TXT_TENSOR'):os.environ.pop(key,None)
        require(os.statvfs(out).f_bavail*os.statvfs(out).f_frsize>=160<<20,'Need160MiB free')
        pin(__file__,r['script_sha256'])
        p19=module(pin('/tmp/vggt-v119-source-r1/run_shared_attention_board.py',R119_SHA),'_v120_frozen_shared_driver_constants')
        parent=module(pin('/tmp/vggt-v118-source-r1/run_native_attention_board.py',p19.PARENT_SHA),'_v120_frozen_native_product')
        abi_path=pin(SRC/'vggt_slam/rknn_linear.py',parent.ABI_SHA);sdk=pin('/usr/lib/librknnrt.so',parent.SDK_SHA)
        pin(SRC/'third_party/vggt/vggt/layers/attention.py',parent.ATTENTION_SHA)
        neon_path=pin(E/'vggt-v56-neon-build-r1/fp16_neon.so',parent.NEON_SHA)
        legacy=pin(SRC/'deploy/rk3588/validation/original_native_gpu/native_erf_gelu_opencl.py',p19.LEGACY_SHA)
        mali=pin(E/'libmali-valhall-g610-g6p0-dummy.so',p19.MALI_SHA)
        bridge_path=pin('/tmp/vggt-v119-source-r1/native_attention_softmax_opencl.py',p19.BRIDGE_SHA)
        kernel=pin('/tmp/vggt-v119-source-r1/attention_softmax.cl',p19.KERNEL_SHA)
        runtime_path=pin(Path(__file__).with_name('shared_attention_runtime.py'),RUNTIME_SHA)
        owned_path=pin(E/'vggt-v100-source-r1/native_heads_runtime.py',OWNED_SHA)
        ep=pin(a.export,a.export_sha256);gp=pin(a.check,a.check_sha256);cp=pin(a.compile,a.compile_sha256)
        exported,gated,compiled=[json.loads(path.read_text()) for path in (ep,gp,cp)]
        require(exported['schema']=='v120_original_native518_DINO0_exact_cuts_ONNX_export'
            and exported['status']=='exported_original_DINO0_exact_cuts_not_FP32_or_NPU_accepted'
            and exported['script_sha256']==EXPORTER_SHA
            and exported['checkpoint']==CHECKPOINT and exported['parent']['sha256']==PARENT_ONNX_SHA
            and exported['source_and_inputs_unchanged'] and not exported['cleanup_errors'],'Exact original DINO0 cut export required')
        proof=exported['proof']
        require(all(proof[k] for k in ('original_node_union_exactly_once','whole_parent_reassembled_protobuf_equal',
            'full_initializer_union_original_protobuf_equal','all14_original_learned_parameters_verified','original_middle_12_nodes_preserved',
            'original_residual_input_preserved_in_suffix','all1041_queries_keys_16_heads_64_dimensions'))
            and proof['original_parent_nodes']==31 and proof['original_parent_initializers']==23,'Complete original graph preservation proof required')
        r['all31_original_nodes_accounted']=True
        for data,schema,status in ((gated,'v120_original_native518_DINO0_cuts_FP32_check','checked_original_DINO0_complete_composition_not_NPU_quality_accepted'),
            (compiled,'v120_original_native518_DINO0_cuts_FP16_compile','compiled_original_DINO0_cuts_not_quality_accepted')):
            require(data['schema']==schema and data['status']==status and data['export_receipt']['sha256']==a.export_sha256
                and data['script_sha256']==CHECKER_SHA and data['exporter_source_sha256']==EXPORTER_SHA
                and data['capture']['sha256']==CAPTURE_SHA and data['original_FP32_gate_passed']
                and data['fixed_gate']==dict(relative_l2_max=1e-5,max_abs_max=1e-3)
                and data['source_and_inputs_unchanged'] and data['affinity_restored'] and data['environment_restored']
                and data['cwd_restored'] and not data['cleanup_errors'],'Linked successful board FP32 check/compile required')
        require(gated['models']==exported['models'] and gated['proof']==compiled['proof']==proof
            and gated['mode']=='check' and compiled['mode']=='compile'
            and set(gated['checks'])=={'parent_vs_original','prefix_QKV_vs_original','middle_attention_vs_original','composed_vs_original','composed_vs_parent'}
            and all(v['fixed_gate_passed'] for v in gated['checks'].values()) and compiled['check_receipt']['sha256']==a.check_sha256
            and compiled['configuration']==CONFIG and compiled['build_options']==BUILD and compiled['sdk_released'],
            'Actual original FP32 composition and nonquantized FP16 compilation required')
        originals={row['key']:row for row in exported['models']}
        require(set(originals)=={'prefix','middle','suffix'} and len(compiled['models'])==2,'Exact three reference cuts and two compiled cuts required')
        expected_bindings={'prefix':(['input'],[SHAPE],['attn.qkv'],[QKV]),
            'suffix':(['input','attention_heads_flatten'],[SHAPE,SHAPE],['output'],[SHAPE])}
        for row in compiled['models']:
            key=row['key'];require(key in expected_bindings and {k:v for k,v in row.items() if k!='rknn'}==originals[key],'Compiled cut identity differs')
            inn,ins,outn,outs=expected_bindings[key]
            for direction,names,shapes in [('input',inn,ins),('output',outn,outs)]:
                require(row[direction+'_bindings']==[dict(index=i,name=name,shape=list(shape),dtype='float32')
                    for i,(name,shape) in enumerate(zip(names,shapes))],'Complete original rank3 cut ABI required')
        require({row['key'] for row in compiled['models']}==set(expected_bindings),'Distinct original prefix and suffix required')
        import numpy as np
        def raw(value):return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()
        def array(base,row):
            path=pin(base/row['file'],row['sha256'],row['bytes']);v=np.load(path,allow_pickle=False)
            require(list(v.shape)==row['shape'] and str(v.dtype)==row['dtype'] and raw(v)==row['raw_sha256'] and np.isfinite(v).all(),'Capture tensor differs');return v
        def save(name,value):
            path=out/name
            with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
            return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),raw_sha256=raw(value),shape=list(value.shape),dtype=str(value.dtype))
        def compare(value,reference):
            v,z=value.astype(np.float64),reference.astype(np.float64);delta=v-z;bad=np.abs(delta)>1e-5+1e-4*np.abs(z)
            return dict(relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(z)),max_abs=float(np.abs(delta).max()),strict_rtol=1e-4,
                strict_atol=1e-5,strict_pass=not bool(bad.any()),outside_count=int(bad.sum()),byte_equal=value.tobytes()==reference.tobytes())
        capture_path=pin(E/'vggt-v100-dino0-intermediates-r1/capture.json',CAPTURE_SHA);capture=json.loads(capture_path.read_text())
        require(capture['checkpoint']==CHECKPOINT and capture['final_comparison']['byte_equal_v99']
            and capture['final_comparison']['raw_sha256_equal_v99'],'Original whole-block reference required')
        x=array(capture_path.parent,capture['artifacts']['input']);reference=array(capture_path.parent,capture['artifacts']['output']);input_raw=raw(x)
        require(x.shape==reference.shape==SHAPE,'Complete original native518 input/output required')
        r.update(original_input=capture['artifacts']['input'],original_output_reference=capture['artifacts']['output'],
            export_sha256=a.export_sha256,check_sha256=a.check_sha256,compile_sha256=a.compile_sha256,
            original_graph_proof=exported['proof'],runtime_sha256=RUNTIME_SHA)
        bc=pin(E/'vggt-v99-dino-compile-r1/result.json',BASE_COMPILE_SHA);baseline=json.loads(bc.read_text())
        require(baseline['status']=='compiled_not_quality_accepted' and baseline['onnx']['sha256']==PARENT_ONNX_SHA and baseline['configuration']==CONFIG
            and baseline['build_options']==BUILD and baseline['sdk_released'] and not baseline['cleanup_errors'],'Original baseline compile differs')
        bm=pin(bc.parent/baseline['rknn']['file'],BASE_MODEL_SHA,26517686)
        O=module(owned_path,'_v120_frozen_owned_IO');abi=module(abi_path,'_v120_typed_native_ABI')
        require((ct.sizeof(abi.RKNNMatmulInfo),ct.sizeof(abi.RKNNMatmulIOAttr),ct.sizeof(abi.RKNNTensorMem))==(64,996,40),'MatMul ABI sizes differ')
        pin(E/'vggt-v99-source-r1/run_native_block_r1.py',O.HELPER_SHA)
        pin(SRC/'deploy/rk3588/validation/run-v12-rknn-block-board.py',O.ABI_SHA)
        counted=parent.Counted(abi._API(str(sdk)));gpu=module(bridge_path,'_v120_frozen_softmax_bridge');runtime=module(runtime_path,'_v120_shared_attention_runtime')
        neon=ct.CDLL(str(neon_path))
        def graph(name,model,digest,inn,ins,outn,outs):
            value=O.OwnedContext.__new__(O.OwnedContext);weighted.append((name,value))
            value.__init__(model,digest,input_names=inn,input_shapes=ins,output_names=outn,output_shapes=outs,core_mask=1,flags=0)
            return value
        r['platform_before']=platform_state();require(r['platform_before']['NPU']['frequency_hz']==1000000000,'External1GHz controller required')
        progress('initializing_fused_baseline');tick=time.perf_counter();original=graph('baseline',bm,BASE_MODEL_SHA,['input'],[SHAPE],['output'],[SHAPE])
        r['initialization_seconds']['baseline']=time.perf_counter()-tick
        progress('initializing_actual_prefix_shared_attention_suffix');tick=time.perf_counter()
        cuts={row['key']:row for row in compiled['models']}
        def cut(key):
            row=cuts[key];path=pin(cp.parent/row['rknn']['file'],row['rknn']['sha256'],row['rknn']['bytes'])
            return graph(key,path,row['rknn']['sha256'],[v['name'] for v in row['input_bindings']],
                [v['shape'] for v in row['input_bindings']],[v['name'] for v in row['output_bindings']],[v['shape'] for v in row['output_bindings']])
        prefix=cut('prefix');suffix=cut('suffix')
        shared=runtime.SharedAttention.__new__(runtime.SharedAttention)
        shared.__init__(parent,gpu,abi,counted,np,neon,legacy=legacy,mali=mali,kernel=kernel)
        r['initialization_seconds']['candidate']=time.perf_counter()-tick
        r['GPU_initial']=copy.deepcopy(shared.initial_bridge_metrics)
        seen=set()
        for index,route in enumerate('ABBAABBA'):
            require(platform_state()['NPU']['frequency_hz']==1000000000,'Frequency controller changed');progress('complete_block_'+str(index)+'_'+route)
            tick=time.perf_counter()
            if route=='A':
                y=original.run([x])[0];stages=dict(fused=copy.deepcopy(original.last_timings))
            else:
                qkv=prefix.run([x])[0];pt=copy.deepcopy(prefix.last_timings)
                attn=shared.run(qkv);at=copy.deepcopy(shared.last_timings)
                y=suffix.run([x,attn])[0];stages=dict(prefix=pt,attention=at,suffix=copy.deepcopy(suffix.last_timings))
            require(y.shape==SHAPE and y.dtype==np.float32 and np.isfinite(y).all(),'Complete finite original block output required')
            elapsed=time.perf_counter()-tick
            if route=='A':require(raw(y)==BASE_OUTPUT_RAW,'Original fused baseline bytes changed')
            row=dict(index=index,route=route,first_for_route=route not in seen,complete_block_seconds=elapsed,stages=stages,
                vs_original_CPU=compare(y,reference),residual_update_vs_original_CPU=compare(y-x,reference-x),output=save('call'+str(index)+'.npy',y))
            if route=='B':
                row.update(actual_prefix_QKV_raw_sha256=raw(qkv),actual_attention_raw_sha256=raw(attn),
                    prefix_QKV_Half_roundtrip_byte_equal=qkv.astype(np.float16).astype(np.float32).tobytes()==qkv.tobytes())
                if route not in seen:
                    row['actual_prefix_QKV']=save('actual-prefix-QKV.npy',qkv);row['actual_attention']=save('actual-attention.npy',attn)
            r['calls'].append(row);seen.add(route)
        require(raw(x)==input_raw,'Original residual input modified')
        r['original_input_unchanged']=True
        r['subsequent_three_means_seconds']={route:statistics.mean(x['complete_block_seconds'] for x in r['calls'] if x['route']==route and not x['first_for_route']) for route in 'AB'}
        r['platform_after']=platform_state();r['status']='completed_original_DINO0_component_ABBA_not_full_model_accepted'
    except BaseException as error:
        r.update(status='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        tick=time.perf_counter()
        if shared is not None:
            try:shared.close()
            except BaseException as error:r['cleanup_errors'].append('Shared attention: '+str(error))
            r['shared_cleanup']=copy.deepcopy(getattr(shared,'cleanup',{}));r['GPU_after_close']=copy.deepcopy(getattr(shared,'final_bridge_metrics',{}))
        r['weighted_resources']={}
        for name,owner in reversed(weighted):
            try:owner.close()
            except BaseException as error:r['cleanup_errors'].append(name+' close: '+str(error))
            try:
                metrics=owner.metrics();r['weighted_resources'][name]=metrics
                if not metrics['resource_counts_balanced'] or metrics['live_contexts'] or metrics['live_output_buffers']:r['cleanup_errors'].append(name+' resource imbalance')
                r['cleanup_errors'].extend(metrics['cleanup_errors'])
            except BaseException as error:r['cleanup_errors'].append(name+' unverified owner: '+str(error))
        r['destroy_seconds']=time.perf_counter()-tick
        if counted is not None:r.update(MatMul_attempts=counted.calls,MatMul_successes=counted.successes)
        if r['status'].startswith('completed'):
            expected=dict(rknn_matmul_create=2,rknn_matmul_set_core_mask=2,rknn_create_mem=6,rknn_matmul_set_io_mem=262,
                rknn_mem_sync=512,rknn_matmul_run=128,rknn_destroy_mem=6,rknn_matmul_destroy=2)
            r['expected_MatMul_successes']=expected
            if counted.calls!=expected or counted.successes!=expected:r['cleanup_errors'].append('Native call ledger differs')
            g=r.get('GPU_after_close',{})
            if not g.get('fully_released') or any(g.get(k)!=64 for k in ('calls_attempted','calls_completed','events_created','events_released')) or g.get('objects_created')!=g.get('objects_released'):
                r['cleanup_errors'].append('GPU complete call/release ledger differs')
            for name,metrics in r['weighted_resources'].items():
                if metrics['runs_completed']!=4 or not metrics['contexts_created']==metrics['contexts_destroyed']==1 or metrics['output_buffers_acquired']!=4:
                    r['cleanup_errors'].append(name+' complete weighted ledger differs')
            r['summed_eight_call_lifecycle_seconds']=sum(r['initialization_seconds'].values())+r['destroy_seconds']+sum(x['complete_block_seconds'] for x in r['calls'])
        changed=[str(path) for path,digest in pins.items() if not path.is_file() or sha(path)!=digest]
        r['source_and_inputs_unchanged']=not changed;r['cleanup_errors'].extend('Pin changed: '+path for path in changed)
        try:
            os.sched_setaffinity(0,affinity);os.environ.clear();os.environ.update(env);signal.signal(signal.SIGTERM,old_term)
            r.update(affinity_restored=set(os.sched_getaffinity(0))==affinity,environment_restored=dict(os.environ)==env)
            require(r['affinity_restored'] and r['environment_restored'],'Process state not restored')
        except BaseException as error:r['cleanup_errors'].append(str(error))
        if r['cleanup_errors']:r['status']='failed_cleanup'
        progress('finished')
    return 0 if r['status'].startswith('completed') else 1


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
