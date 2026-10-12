#!/usr/bin/env python3
"""RK3588-only original pair0+pair1 versus their four-output FP16 group.

Both paths include explicit NEON half conversion, owned copies and finite
checks. No profile flag, warmup hidden from results, model substitution or
system clock write. The complete original FP32/SLAM gates remain separate.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
SDK_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
NEON_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
SHAPE = (2, 53, 1024)
ORDER = ('A', 'B', 'B', 'A', 'A', 'B', 'B', 'A')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()

def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    if not re.fullmatch('[0-9a-f]{64}', expected) or sha(path) != expected:
        raise ValueError('Frozen artifact changed: '+str(path))
    pins[path] = expected
    return path

def artifact(root, row, pins):
    root = Path(root).resolve(strict=True)
    path = (root / row['file']).resolve(strict=True)
    if not path.is_relative_to(root) or path.stat().st_size != row['bytes']:
        raise ValueError('Artifact escapes directory or size differs')
    return pin(path, row['sha256'], pins)

def compare(np, x, y):
    if x.shape != y.shape or x.dtype != y.dtype or not np.isfinite(y).all():
        raise ValueError('Reference shape/dtype/finite mismatch')
    delta = x.astype(np.float64)-y.astype(np.float64)
    failed = np.abs(delta) > 1e-5 + 1e-4*np.abs(y.astype(np.float64))
    denominator = float(np.linalg.norm(y.astype(np.float64)))
    error = float(np.linalg.norm(delta))
    return dict(bitwise_equal=x.tobytes()==y.tobytes(), max_abs=float(np.abs(delta).max()),
                relative_l2=error/denominator if denominator else (0. if not error else None),
                strict_rtol=1e-4, strict_atol=1e-5, failed_elements=int(failed.sum()),
                strict_pass=not bool(failed.any()))

def sample():
    n = Path('/sys/class/devfreq/fdab0000.npu')
    return dict(actual_frequency_hz=int((n/'cur_freq').read_text()),
                governor=(n/'governor').read_text().strip(),
                temperatures_c={p.joinpath('type').read_text().strip():
                    int(p.joinpath('temp').read_text())/1000
                    for p in Path('/sys/class/thermal').glob('thermal_zone*')})

def execute(base, api, neon, np, context, array, bindings, counters, cleanup_errors):
    started = time.perf_counter()
    if array.shape != SHAPE or array.dtype != np.float32 or not array.flags.c_contiguous or not np.isfinite(array).all():
        raise ValueError('Dense finite original frame input required')
    feed = np.empty(SHAPE, dtype=np.float16)
    neon.f32_to_f16(array.ctypes.data, feed.ctypes.data, array.size)
    if not np.isfinite(feed).all(): raise ValueError('FP16 conversion overflow')
    inputs = (base.Input*1)(base.Input(0, feed.ctypes.data, feed.nbytes, 0, 1, 3))
    count = len(bindings)
    outputs = (base.Output*count)()
    for i in range(count): outputs[i].index, outputs[i].want_float = i, 0
    run, received = base.RunExtend(0, 0, 30000, -1), base.OutputExtend()
    got, halves, timing = False, [], {}
    def call(name, fn, *args):
        begin = time.perf_counter()
        try: return fn(*args)
        finally: timing[name+'_seconds'] = time.perf_counter()-begin
    try:
        base.check_call(call('inputs_set', api.rknn_inputs_set, context, 1, inputs), 'rknn_inputs_set')
        counters['sdk_run_attempts'] += 1
        code = call('run', api.rknn_run, context, ct.byref(run))
        counters['sdk_runs_succeeded'] += int(code == 0)
        base.check_call(code, 'rknn_run')
        code = call('outputs_get', api.rknn_outputs_get, context, count, outputs, ct.byref(received))
        counters['outputs_acquired'] += sum(bool(o.buf) for o in outputs)
        base.check_call(code, 'rknn_outputs_get'); got = True
        if run.frame_id != received.frame_id: raise ValueError('Synchronous frame ID mismatch')
        for output, binding in zip(outputs, bindings):
            shape = tuple(binding['shape']); size = math.prod(shape)
            if not output.buf or output.size != size*2: raise ValueError('Logical half output pointer/size differs')
            view = np.ctypeslib.as_array(ct.cast(output.buf, ct.POINTER(ct.c_uint16)), shape=(size,)).view(np.float16)
            value = np.empty(shape, dtype=np.float16)
            np.copyto(value.reshape(-1), view); halves.append(value)
    finally:
        if got or any(o.buf for o in outputs):
            acquired = sum(bool(o.buf) for o in outputs)
            code = call('outputs_release', api.rknn_outputs_release, context, count, outputs)
            if code == 0: counters['outputs_released'] += acquired
            else: cleanup_errors.append('rknn_outputs_release returned '+str(code))
            base.check_call(code, 'rknn_outputs_release')
    owned = []
    for half in halves:
        value = np.empty(half.shape, dtype=np.float32)
        neon.f16_to_f32(half.ctypes.data, value.ctypes.data, half.size)
        if not value.flags.owndata or not np.isfinite(value).all(): raise ValueError('Finite owned output required')
        owned.append(value)
    timing.update(route_seconds=time.perf_counter()-started, output_count=count,
                  input_api_type=1, output_want_float=0, outputs_released=True)
    return owned, timing

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('manifest','compile-result','check-result','output-dir'):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ('manifest-sha256','compile-result-sha256','check-result-sha256'):
        p.add_argument('--'+n,required=True)
    a = p.parse_args()
    if platform.machine() != 'aarch64' or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        p.error('Actual SDK execution only on RK3588')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    pins, contexts, api, base = {}, [], None, None
    old_affinity = os.sched_getaffinity(0)
    report = dict(schema='v64_original_two_pair_group_FP16_IO_probe',status='failed',
                  script_sha256=sha(__file__),original_model_only=True,candidate_accepted=False,
                  full_model_or_trajectory_or_realtime_validated=False,normal_init_flags=0,
                  core_mask=7,order=list(ORDER),hidden_warmup_calls=0,calls=[],queries=[],
                  contexts_created=0,contexts_destroyed=0,cleanup_errors=[],
                  counters=dict(sdk_run_attempts=0,sdk_runs_succeeded=0,outputs_acquired=0,outputs_released=0),
                  timing_scope='Two original pair SDK routes vs one complete four-output group, including NEON FP16 I/O, finite checks and owned copies; excludes hash/NPZ save; no profiler',
                  system_clocks_changed=False)
    try:
        os.sched_setaffinity(0,{4,5,6,7});report['cpu_affinity']=[4,5,6,7]
        import numpy as np
        root=BASE/'src'; parent_path=pin(BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json',PARENT_SHA,pins)
        parent=json.loads(parent_path.read_text())
        export=json.loads(pin(a.manifest,a.manifest_sha256,pins).read_text())
        compiled=json.loads(pin(a.compile_result,a.compile_result_sha256,pins).read_text())
        checked=json.loads(pin(a.check_result,a.check_result_sha256,pins).read_text())
        if (export['schema']!='v64_original_VGGT_pair0_pair1_group_ONNX_manifest' or len(export['blocks'])!=1
                or export['checkpoint']!=parent['checkpoint'] or export['original_parent_manifest_sha256']!=PARENT_SHA):
            raise ValueError('Only fixed complete original pair0+pair1 group supported')
        group=export['blocks'][0]
        if group['parents']!=parent['blocks'][:2] or group['indices']!=[0,1] or group['input_shape']!=list(SHAPE):
            raise ValueError('Frozen original pair lineage differs')
        expected=[dict(index=i,name=f'pair{i//2}_{role}_output',shape=list(shape),dtype='float32')
                  for i,(role,shape) in enumerate([('frame',SHAPE),('global',(1,106,1024)),('frame',SHAPE),('global',(1,106,1024))])]
        if group['output_bindings']!=expected: raise ValueError('All four original outputs required')
        if compiled['manifest_sha256']!=a.manifest_sha256 or compiled['check_receipt_sha256']!=a.check_result_sha256 or compiled['cleanup_errors']:
            raise ValueError('Compile/check provenance differs')
        config=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,
                    mean_values=None,std_values=None,quant_img_RGB2BGR=False,model_pruning=False,
                    remove_reshape=False,quantize_weight=False,enable_flash_attention=False)
        if (compiled['status']!='compiled_not_quality_accepted' or compiled['configuration']!=config
                or [x['name'] for x in compiled['operations']]!=['config','load_onnx','build','export_rknn']
                or any(x['code'] not in (None,0) for x in compiled['operations'])):
            raise ValueError('Original nonquantized FP16 compile configuration differs')
        if checked['manifest_sha256']!=a.manifest_sha256 or not checked['all_separate_ONNX_checks_byte_equal']:
            raise ValueError('Original graph-chain verification required')
        for relative,digest in parent['source_hashes'].items():
            path=(root/relative).resolve(strict=True)
            if not path.is_relative_to(root.resolve()):raise ValueError('Source escaped root')
            pin(path,digest,pins)
        abi_path=pin(root/'deploy/rk3588/validation/run-v12-rknn-block-board.py',ABI_SHA,pins)
        pin(root/'deploy/rk3588/reference/rknn_api_v2.3.2.h',HEADER_SHA,pins)
        spec=importlib.util.spec_from_file_location('_v64_ABI',abi_path)
        base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        report['board']=base.require_board()
        api=base.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so',SDK_SHA,pins))))
        neon=ct.CDLL(str(pin(BASE/'experiments/vggt-v56-neon-build-r1/fp16_neon.so',NEON_SHA,pins)))
        for name in ('f32_to_f16','f16_to_f32'):
            fn=getattr(neon,name);fn.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];fn.restype=None
        artifact(a.manifest.parent,group,pins)
        models=[('A0',artifact(parent_path.parent,parent['blocks'][0]['rknn'],pins),parent['blocks'][0]['output_bindings']),
                ('A1',artifact(parent_path.parent,parent['blocks'][1]['rknn'],pins),parent['blocks'][1]['output_bindings']),
                ('B',artifact(a.compile_result.parent,compiled['rknn'],pins),expected)]
        routes={}
        def descriptor(attr,shape,name,index):
            r=base.attr_record(attr)
            identity=r['quantization_type']==2 and r['zero_point']==0 and r['scale']==1.
            if (r['index']!=index or r['name']!=name or r['shape']!=list(shape) or r['type']!=1
                    or r['format']!=3 or r['pass_through']!=0 or r['n_elems']!=math.prod(shape)
                    or r['size']!=2*math.prod(shape) or (r['quantization_type']!=0 and not identity)):
                raise ValueError('Dense original half descriptor differs: '+str(r))
            return r
        for name,path,bindings in models:
            context=base.CONTEXT();contexts.append(context)
            filename=ct.create_string_buffer(str(path).encode())
            code=api.rknn_init(ct.byref(context),ct.cast(filename,ct.c_void_p),0,0,None)
            report['contexts_created']+=int(bool(context.value));base.check_call(code,'rknn_init')
            if not context.value:raise ValueError('Empty NPU context')
            base.check_call(api.rknn_set_core_mask(context,7),'rknn_set_core_mask')
            version=base.query(api,context,5,base.SDKVersion())
            if not re.match(r'^2\.3\.2(?:\D|$)',bytes(version.api_version).decode()):raise ValueError('SDK2.3.2 required')
            count=base.query(api,context,0,base.InputOutputNum())
            if (count.n_input,count.n_output)!=(1,len(bindings)):raise ValueError('Input/output count differs')
            inp=descriptor(base.query(api,context,1,base.TensorAttr(index=0)),SHAPE,'input',0)
            outs=[descriptor(base.query(api,context,2,base.TensorAttr(index=i)),b['shape'],b['name'],i) for i,b in enumerate(bindings)]
            report['queries'].append(dict(role=name,input=inp,outputs=outs,model_sha256=sha(path)))
            routes[name]=(context,bindings)
        report.update(parent_manifest_sha256=PARENT_SHA,export_manifest_sha256=a.manifest_sha256,
                      compile_result_sha256=a.compile_result_sha256,check_result_sha256=a.check_result_sha256)
        for case in checked['cases']:
            array=np.load(artifact(case['directory'],case['input'],pins),allow_pickle=False)
            references=([np.load(artifact(case['directory'],r,pins),allow_pickle=False) for r in case['original_outputs']]
                        if case.get('original_outputs') else None)
            array_digest=hashlib.sha256(array.tobytes()).hexdigest()
            first={}
            for sequence,role in enumerate(ORDER):
                before=sample();start=time.perf_counter();timings=[]
                values=[];feed=array
                for key in (('A0','A1') if role=='A' else ('B',)):
                    ctx,bindings=routes[key]
                    out,timing=execute(base,api,neon,np,ctx,feed,bindings,report['counters'],report['cleanup_errors'])
                    values.extend(out);timings.append(timing);feed=out[-1].reshape(SHAPE)
                elapsed=time.perf_counter()-start;after=sample()
                cold=role not in first
                if cold:first[role]=values
                path=a.output_dir/f"{case['tag']}-{sequence:02d}-{role}.npz"
                with path.open('xb') as stream:np.savez(stream,**{f'output{i}':v for i,v in enumerate(values)})
                report['calls'].append(dict(input=case['tag'],sequence=sequence,role=role,first_for_input_role=cold,
                    component_seconds=elapsed,routes=timings,platform_before=before,platform_after=after,
                    original_FP32=[compare(np,x,y) for x,y in zip(values,references)] if references else None,
                    same_input_first_A=[compare(np,x,y) for x,y in zip(values,first['A'])],
                    repeat_first_role=[compare(np,x,y) for x,y in zip(values,first[role])],
                    artifact=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))))
            if hashlib.sha256(array.tobytes()).hexdigest()!=array_digest:raise ValueError('Original input mutated')
        n=len(checked['cases'])
        if report['counters']!=dict(sdk_run_attempts=12*n,sdk_runs_succeeded=12*n,outputs_acquired=32*n,outputs_released=32*n):
            raise ValueError('All expected SDK calls/output releases required')
        report['summary']={}
        for role in ('A','B'):
            rows=[x for x in report['calls'] if x['role']==role and not x['first_for_input_role']]
            report['summary'][role]=dict(excluding_first_per_input_count=len(rows),mean_component_seconds=sum(x['component_seconds'] for x in rows)/len(rows))
        report['status']='completed_component_diagnostic_not_adopted'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        for context in reversed(contexts):
            if context.value and api is not None:
                try:
                    base.check_call(api.rknn_destroy(context),'rknn_destroy')
                    context.value=0;report['contexts_destroyed']+=1
                except BaseException as exc:report['cleanup_errors'].append(str(exc))
        report['live_contexts']=sum(bool(x.value) for x in contexts)
        try:
            for path,digest in pins.items():
                if sha(path)!=digest:raise ValueError('Frozen evidence mutated: '+str(path))
            os.sched_setaffinity(0,old_affinity)
            report['frozen_pins_unchanged']=True
        except BaseException as exc:report['cleanup_errors'].append(str(exc))
        if report['cleanup_errors']:report['status']='failed_cleanup'
        (a.output_dir/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','summary','error','contexts_created','contexts_destroyed','cleanup_errors')}),flush=True)
    if report['status']!='completed_component_diagnostic_not_adopted':raise SystemExit(1)

if __name__=='__main__':main()
