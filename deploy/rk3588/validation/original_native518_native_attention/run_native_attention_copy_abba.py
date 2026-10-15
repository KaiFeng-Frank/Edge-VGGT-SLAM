#!/usr/bin/env python3
"""Original DINO0 attention: ABBA of QK native-C read/copy order only.

Two native-layout persistent contexts, all 16 heads and 1041 keys. Alignment
padding never participates in softmax. Dynamic B upload is timed on every head.
FP16 probability storage is explicit; FP32 output does not prove accumulation
precision. This is an operator screen, not complete-model acceptance.
"""
import argparse
import ctypes as ct
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import sys
import time
import traceback

B = Path('/mnt/vggt-data/vggt-slam2')
E = B / 'experiments'
TAP_SHA = '1f89eac69b85ec84ea76f9e8ca0a50ca7d4c5f19620c4159f41eaf6f3661225a'
CF_SHA = '03e9a5797ca21c542e467a68057f7290f0a881a689b273db74f7f7d5d3dea94d'
DUMP_SHA = '565054e051afd2565bcd07d6487e02cedeca62456e09e1214021f30c9c4f46d4'
QKV_RAW = '1f09065df446dbbc56ca249b96156f7ec2c78b87555461c3ed33852e40f91936'
ABI_SHA = '69056a2cb44ba69a2d311265b71ff919e4758fcaa150834a0b0cfe85dbcaef7d'
SDK_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
NEON_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
ATTENTION_SHA = '9d80ff6c6f562efc29d806125fe47bd9b504a6356ce2d9b5d61e8ee4ea770a00'
N, PAD, H, D = 1041, 1056, 16, 64


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class Counted:
    def __init__(self, api):
        self.api, self.calls, self.successes = api, {}, {}

    def __getattr__(self, name):
        f = getattr(self.api, name)
        def call(*args):
            self.calls[name] = self.calls.get(name, 0) + 1
            value = f(*args)
            success = bool(value) if name == 'rknn_create_mem' else value == 0
            if success:
                self.successes[name] = self.successes.get(name, 0) + 1
            return value
        return call


class Product:
    """Own one context and its A/B/C buffers, including partial-init cleanup."""
    def __init__(self, abi, api, np, k, n):
        self.abi, self.api, self.np, self.k, self.n = abi, api, np, k, n
        self.ctx, self.attrs = abi.RKNNContext(), abi.RKNNMatmulIOAttr()
        self.memories, self.views, self.descriptors, self.cleanup_errors = {}, {}, {}, []
        self.copy_native_first = False

    def open(self):
        abi, api, np = self.abi, self.api, self.np
        info = abi.RKNNMatmulInfo(M=N, K=self.k, N=self.n,
            type=abi.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, B_layout=1, AC_layout=1)
        abi._check(api.rknn_matmul_create(ct.byref(self.ctx), ct.byref(info), ct.byref(self.attrs)), 'create')
        require(self.ctx.value != 0, 'Null successful context')
        abi._check(api.rknn_matmul_set_core_mask(self.ctx, 1), 'core1')
        dims = dict(A=(self.k // 8, N, 8), B=(self.n // 16, self.k // 32, 16, 32), C=(self.n // 4, N, 4))
        for name in 'ABC':
            attr = getattr(self.attrs, name)
            shape = tuple(attr.dims[:attr.n_dims])
            dtype = np.float32 if name == 'C' else np.float16
            require(shape == dims[name] and attr.type == (0 if name == 'C' else 1)
                and attr.size >= math.prod(shape) * np.dtype(dtype).itemsize,
                'Unexpected native ' + name + ' descriptor: ' + str(shape))
            self.descriptors[name] = dict(shape=list(shape), type=int(attr.type), bytes=int(attr.size))
            ptr = api.rknn_create_mem(self.ctx, attr.size)
            require(bool(ptr), 'Allocation failed: ' + name)
            self.memories[name] = ptr
            require(ptr.contents.virt_addr and ptr.contents.size >= attr.size, 'Invalid allocation: ' + name)
            if name != 'C':
                ct.memset(ptr.contents.virt_addr, 0, attr.size)
            raw = (ct.c_uint8 * attr.size).from_address(ptr.contents.virt_addr)
            self.views[name] = np.frombuffer(raw, dtype=dtype, count=math.prod(shape)).reshape(shape)
            self.bind(name)

    def bind(self, name):
        self.abi._check(self.api.rknn_matmul_set_io_mem(self.ctx, self.memories[name],
            ct.byref(getattr(self.attrs, name))), 'bind ' + name)

    def sync(self, name, direction):
        self.abi._check(self.api.rknn_mem_sync(self.ctx, self.memories[name], direction), 'sync ' + name)

    def run(self, a, b):
        np = self.np
        require(a.shape == (N, self.k) and b.shape == (self.k, self.n)
            and a.dtype == b.dtype == np.float16, 'Product logical inputs differ')
        t0 = time.perf_counter()
        av = a.reshape(N, self.k // 8, 8).transpose(1, 0, 2)
        bv = b.reshape(self.k // 32, 32, self.n // 16, 16).transpose(2, 0, 3, 1)
        np.copyto(self.views['A'], av, casting='no')
        np.copyto(self.views['B'], bv, casting='no')
        self.sync('A', self.abi.RKNN_MEMORY_SYNC_TO_DEVICE)
        self.sync('B', self.abi.RKNN_MEMORY_SYNC_TO_DEVICE)
        self.bind('A')
        self.bind('B')
        t1 = time.perf_counter()
        self.abi._check(self.api.rknn_matmul_run(self.ctx), 'run')
        t2 = time.perf_counter()
        self.sync('C', self.abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
        if self.copy_native_first:
            native = np.array(self.views['C'], copy=True, order='C')
            c = np.ascontiguousarray(native.transpose(1, 0, 2)).reshape(N, self.n)
        else:
            c = np.array(self.views['C'].transpose(1, 0, 2), copy=True, order='C').reshape(N, self.n)
        require(np.isfinite(c).all(), 'Nonfinite product')
        t3 = time.perf_counter()
        return c, dict(pack_upload_bind_seconds=t1-t0, SDK_run_seconds=t2-t1,
                       sync_owned_output_finite_seconds=t3-t2, complete_product_seconds=t3-t0)

    def close(self):
        self.views.clear()
        for name, pointer in list(self.memories.items())[::-1]:
            code = self.api.rknn_destroy_mem(self.ctx, pointer)
            if code:
                self.cleanup_errors.append('destroy_mem ' + name + ': ' + str(code))
            else:
                del self.memories[name]
        if self.ctx.value and not self.memories:
            code = self.api.rknn_matmul_destroy(self.ctx)
            if code:
                self.cleanup_errors.append('destroy: ' + str(code))
            else:
                self.ctx.value = 0
        return dict(K=self.k, output_N=self.n, descriptors=self.descriptors,
            live_context=int(bool(self.ctx.value)), live_memories=len(self.memories), cleanup_errors=self.cleanup_errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('/tmp/vggt-v118-copy-abba-r1'))
    p.add_argument('--source-root', type=Path, default=B / 'src')
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S execution only')
    out = a.output_dir.resolve()
    require(str(out).startswith('/tmp/'), 'Private fresh /tmp destination required')
    out.mkdir(parents=True, exist_ok=False)
    faulthandler.enable(all_threads=True)
    old_env, old_affinity = dict(os.environ), set(os.sched_getaffinity(0))
    old_term = signal.getsignal(signal.SIGTERM)
    contexts, pins, counted = [], {}, None
    began = time.perf_counter()
    report = dict(schema='v118_original_actual_QKV_native_attention_QK_copy_ABBA', status='running', script_sha256=sha(__file__),
        parent_source_sha256='6302f2f2efa236b317ce5985923a21897945906b1d68cd21006c9422618c92d2',
        comparison_routes={'A':'direct strided nativeC read to normal owned array', 'B':'contiguous nativeC owned copy then cached transpose'},
        changed_boundary='QK C read/copy order only; PV and all arithmetic unchanged',
        original_model_only=True, source_QKV_shape=[1,N,3*H*D], original_heads=H, original_keys=N,
        padded_storage_keys=PAD, padding_excluded_from_softmax=True, SDK_type=1, core_mask=1, AC_layout=1, B_layout=1,
        softmax='stable FP32 NumPy on exactly1041 columns, original scale0.125; probability stored in Half for PV',
        original_checkpoint_or_block_deserialized=False, CPU_reference_NN_executed=False, ORT_or_GPU_executed=False,
        calls=[], cleanup_errors=[], candidate_accepted=False, full_model_or_SLAM_or_realtime_validated=False,
        frequency_written=False, init_seconds=None, destroy_seconds=None,
        timing_scope='Each complete attention call includes QKV Half conversion, all16 heads, both dynamic B packing/upload/sync/bind operations per head, SDK execution, output sync/copy, crop/scale/FP32 softmax/Half probability storage, final head merge and finite checks. Context setup/destroy separate; comparisons/hashes/saving excluded.')
    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-began, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic(out/'result.json', report)
        print(json.dumps(dict(stage=stage)), flush=True)
    def pin(path, digest):
        path = Path(path).resolve(strict=True)
        require(sha(path) == digest, 'Pin differs: ' + str(path))
        pins[path] = digest
        return path
    def term(number, frame):
        raise InterruptedError('External timeout/temperature controller terminated worker')
    signal.signal(signal.SIGTERM, term)
    try:
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', RKNN_LOG_LEVEL='0')
        for key in ('RKNN_DUMP_DIR','RKNN_DUMP_QUANT','RKNN_DUMP_TXT_TENSOR'):
            os.environ.pop(key,None)
        require({4,5,6,7} <= old_affinity, 'A76 CPU availability differs')
        os.sched_setaffinity(0, {4,5,6,7})
        require(os.statvfs(out).f_bavail * os.statvfs(out).f_frsize >= 128 << 20, 'Need128MiB free')
        pin(__file__, report['script_sha256'])
        pin(Path(__file__).parent/'run_native_attention_board.py', report['parent_source_sha256'])
        pin(a.source_root/'third_party/vggt/vggt/layers/attention.py', ATTENTION_SHA)
        abi_path = pin(a.source_root/'vggt_slam/rknn_linear.py', ABI_SHA)
        sdk = pin('/usr/lib/librknnrt.so', SDK_SHA)
        neon_path = pin(E/'vggt-v56-neon-build-r1/fp16_neon.so', NEON_SHA)
        tp = pin(E/'vggt-v100-dino-taps-core1-r1/result.json', TAP_SHA)
        cf = pin(E/'vggt-v100-qkv-counterfactual-r1/result.json', CF_SHA)
        dr = pin('/tmp/vggt-v117-dump-r1/result.json', DUMP_SHA)
        t, f, drj = [json.loads(x.read_text()) for x in (tp,cf,dr)]
        require(t['final_baseline_byte_equal_all_calls'] and f['baseline_all_points_byte_equal_original_CPU']
            and f['QKV_substitution_calls']==1 and drj['final_baseline_byte_equal'] and not drj['cleanup_errors'], 'Prior actual evidence gates fail')
        import numpy as np
        def raw(v):
            return hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()
        def load(base,row):
            path=pin(base/row['file'],row['sha256']); z=np.load(path,allow_pickle=False)
            require(z.dtype==np.float32 and raw(z)==row['raw_sha256'] and np.isfinite(z).all(), 'Invalid reference tensor')
            return z
        row=t['calls'][0]['artifact']
        path=pin(tp.parent/row['file'],row['sha256'])
        with np.load(path,allow_pickle=False) as z:
            qkv=np.ascontiguousarray(z['attn.qkv'])
        require(qkv.shape==(1,N,3*H*D) and qkv.dtype==np.float32 and raw(qkv)==QKV_RAW
            and qkv.astype(np.float16).astype(np.float32).tobytes()==qkv.tobytes(), 'Exact Half-representable actual QKV required')
        reference=load(cf.parent,f['artifacts']['attn.proj.input'])
        drow=next(x for x in drj['dump_files'] if x['file'].startswith('008_'))
        fused=load(dr.parent/'dumps',drow)
        fused=np.ascontiguousarray(fused[:,:,0,:].transpose(0,2,1))
        require(reference.shape==fused.shape==(1,N,H*D), 'Original output shape differs')
        report.update(checkpoint=f['checkpoint'], QKV_raw_sha256=QKV_RAW, same_QKV_CPU_reference_raw_sha256=raw(reference),
                      unchanged_fused_reference_raw_sha256=raw(fused), QKV_Half_roundtrip_byte_equal=True)
        spec=importlib.util.spec_from_file_location('_v118_pinned_matmul_ABI',abi_path)
        abi=importlib.util.module_from_spec(spec);spec.loader.exec_module(abi)
        require((ct.sizeof(abi.RKNNMatmulInfo),ct.sizeof(abi.RKNNMatmulIOAttr),ct.sizeof(abi.RKNNTensorMem))==(64,996,40),'ABI sizes differ')
        counted=Counted(abi._API(str(sdk)))
        neon=ct.CDLL(str(neon_path));neon.f32_to_f16.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];neon.f32_to_f16.restype=None
        def half(v):
            v=np.ascontiguousarray(v,dtype=np.float32); h=np.empty(v.shape,np.float16)
            neon.f32_to_f16(v.ctypes.data,h.ctypes.data,v.size)
            require(np.isfinite(h).all(),'Half overflow')
            return h
        def compare(y,ref):
            yd,rd=y.astype(np.float64),ref.astype(np.float64); d=yd-rd
            bad=np.abs(d)>1e-5+1e-4*np.abs(rd)
            return dict(relative_l2=float(np.linalg.norm(d)/np.linalg.norm(rd)),max_abs=float(np.abs(d).max()),
                strict_rtol=1e-4,strict_atol=1e-5,strict_pass=not bool(bad.any()),outside_count=int(bad.sum()))
        progress('opening_two_native_contexts')
        start=time.perf_counter()
        for k,n in ((D,PAD),(PAD,D)):
            product=Product(abi,counted,np,k,n);contexts.append(product);product.open()
        report['init_seconds']=time.perf_counter()-start
        qk,pv=contexts
        first_seen=set()
        for iteration,route in enumerate('ABBAABBA'):
            qk.copy_native_first = route=='B'
            require(int(Path('/sys/class/devfreq/fdab0000.npu/cur_freq').read_text())==1000000000,'External1GHz controller required')
            progress('attention_call_'+str(iteration))
            start=time.perf_counter()
            values=half(qkv).reshape(1,N,3,H,D).transpose(2,0,3,1,4)
            q,k,v=values[:,0]
            key=np.zeros((PAD,D),np.float16);value=np.zeros((PAD,D),np.float16);prob=np.zeros((N,PAD),np.float16)
            yheads=np.empty((H,N,D),np.float32); heads=[]
            for head in range(H):
                key[:N]=k[head];value[:N]=v[head]
                scores,qt=qk.run(q[head],key.T)
                tick=time.perf_counter()
                logits=np.ascontiguousarray(scores[:,:N]);logits*=np.float32(0.125)
                logits-=logits.max(axis=1,keepdims=True)
                np.exp(logits,out=logits)
                logits/=logits.sum(axis=1,keepdims=True,dtype=np.float32)
                prob[:,:N]=half(logits)
                require(not np.any(prob[:,N:]) and not np.any(value[N:]),'Storage padding must remain zero')
                soft=time.perf_counter()-tick
                yheads[head],pt=pv.run(prob,value)
                heads.append(dict(head=head,QK=qt,softmax_crop_scale_probability_store_seconds=soft,PV=pt))
            y=np.ascontiguousarray(yheads.transpose(1,0,2).reshape(1,N,H*D))
            require(np.isfinite(y).all(),'Nonfinite attention output')
            elapsed=time.perf_counter()-start
            path=out/('call'+str(iteration)+'.npy')
            with path.open('xb') as stream:np.save(stream,y,allow_pickle=False)
            require(raw(y)=='00545f97739c7b0a0748918cf766fcd808a9060e2edf6772ec0e88ed502e4fff', 'Every route must byte-match the frozen actual parent output')
            report['calls'].append(dict(index=iteration,route=route,first_call_for_route=route not in first_seen,
                same_contexts_shared_by_all_calls=True,complete_attention_seconds=elapsed,heads=heads,
                vs_original_CPU_same_QKV=compare(y,reference),vs_unchanged_fused_same_QKV=compare(y,fused),
                output=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),raw_sha256=raw(y),shape=list(y.shape),dtype=str(y.dtype))))
            first_seen.add(route)
        report['subsequent_three_means_seconds']={route:statistics.mean(c['complete_attention_seconds'] for c in report['calls']
            if c['route']==route and not c['first_call_for_route']) for route in 'AB'}
        report['status']='completed_component_screen_not_full_model_accepted'
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        tick=time.perf_counter();resources=[]
        for context in reversed(contexts):
            try:
                item=context.close();resources.append(item)
                report['cleanup_errors'].extend(item['cleanup_errors'])
            except BaseException as exc:
                report['cleanup_errors'].append(str(exc))
        report.update(destroy_seconds=time.perf_counter()-tick,resources=resources)
        if report['init_seconds'] is not None and len(report['calls']) == 8:
            report['complete_eight_call_lifecycle_seconds'] = (report['init_seconds'] + report['destroy_seconds']
                + sum(row['complete_attention_seconds'] for row in report['calls']))
        if counted is not None:
            report.update(SDK_attempts=counted.calls,SDK_successes=counted.successes)
            s=counted.successes
            report['resource_counts_balanced']=(s.get('rknn_matmul_create',0)==s.get('rknn_matmul_destroy',0)
                and s.get('rknn_create_mem',0)==s.get('rknn_destroy_mem',0)
                and all(r['live_context']==r['live_memories']==0 for r in resources))
            if not report['resource_counts_balanced']:report['cleanup_errors'].append('Unbalanced native resources')
            if report['status'].startswith('completed') and s.get('rknn_matmul_run',0)!=256:
                report['cleanup_errors'].append('Expected256 actual full-attention products')
        for path,digest in pins.items():
            if not path.is_file() or sha(path)!=digest:report['cleanup_errors'].append('Pin changed: '+str(path))
        report['source_and_inputs_unchanged']=not report['cleanup_errors']
        try:
            os.sched_setaffinity(0,old_affinity);os.environ.clear();os.environ.update(old_env);signal.signal(signal.SIGTERM,old_term)
            report.update(affinity_restored=set(os.sched_getaffinity(0))==old_affinity,environment_restored=dict(os.environ)==old_env)
            require(report['affinity_restored'] and report['environment_restored'],'Process state restoration failed')
        except BaseException as exc:report['cleanup_errors'].append(str(exc))
        if report['cleanup_errors']:report['status']='failed_cleanup'
        progress('finished')
    return 0 if report['status'].startswith('completed') else 1


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
