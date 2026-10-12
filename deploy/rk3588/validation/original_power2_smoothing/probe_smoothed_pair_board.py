#!/usr/bin/env python3
"""Original pair: FP16, calibrated W8A8, power-two-smoothed W8A8; same FP32 API."""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

BASE=Path('/mnt/vggt-data/vggt-slam2')
PARENT_SHA='1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
QUANT_SHA='bdfe3757a7b4adef891dd622525d8a70ba948e9c0af9d0c1d9fffed41a74ea91'
GRAPH_SHA='efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
ABI_SHA='37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
SDK_SHA='d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
HEADER_SHA='c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
CHECK_SOURCE_SHA='896ee7f99cf3b6d699e9c20826e618a696f3f54600705318a3054778d4f89615'
COMPILE_SOURCE_SHA='c0caf28ec34c0b323799e8030e758f606a8ef39df182d6f6ebe5a980cc5d1626'
HELPER_SHA='4094e393d0504954a67993c24e82c5534862393930f60db3ec64c2c3e4098b76'
SHAPES=([2,53,1024],[1,106,1024])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for x in iter(lambda:stream.read(4<<20),b''):h.update(x)
    return h.hexdigest()


def sample():
    return dict(actual_frequency_hz=int(Path('/sys/class/devfreq/fdab0000.npu/cur_freq').read_text()),
        temperatures_c={(p/'type').read_text().strip():int((p/'temp').read_text())/1000
            for p in Path('/sys/class/thermal').glob('thermal_zone*')})


def load(path,digest,name,pins):
    assert sha(path)==digest; pins[str(path)]=digest
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def io_desc(base,attr,shape,name,index):
    r=base.attr_record(attr);identity=r['type']==1 and r['quantization_type']==2 and r['zero_point']==0 and r['scale']==1.
    assert r['index']==index and r['name']==name and r['shape']==shape and r['n_elems']==math.prod(shape)
    assert r['type'] in (0,1,2,3) and r['format']==3 and r['pass_through']==0 and math.isfinite(r['scale'])
    assert r['size']==math.prod(shape)*{0:4,1:2,2:1,3:1}[r['type']]
    if r['type'] in (2,3):
        assert r['quantization_type']==2 and r['scale']>0 and (-128<=r['zero_point']<=127 if r['type']==2 else 0<=r['zero_point']<=255)
    else:assert r['quantization_type']==0 or identity
    return r


def execute(base,api,np,context,x,counters,errors):
    inputs=(base.Input*1)(base.Input(0,x.ctypes.data,x.nbytes,0,0,3))
    outputs=(base.Output*2)()
    for i in range(2):outputs[i].index=i;outputs[i].want_float=1
    run,received=base.RunExtend(0,0,30000,-1),base.OutputExtend()
    got=False;timing={};start=time.perf_counter()
    try:
        begin=time.perf_counter();base.check_call(api.rknn_inputs_set(context,1,inputs),'inputs_set');timing['inputs_set']=time.perf_counter()-begin
        begin=time.perf_counter();counters['run_attempts']+=1;code=api.rknn_run(context,ct.byref(run))
        counters['runs_succeeded']+=int(code==0);base.check_call(code,'run');timing['run']=time.perf_counter()-begin
        begin=time.perf_counter();base.check_call(api.rknn_outputs_get(context,2,outputs,ct.byref(received)),'outputs_get')
        got=True;counters['outputs_acquired']+=2;timing['outputs_get']=time.perf_counter()-begin
        assert run.frame_id==received.frame_id,'Synchronous frame ID differs'
        begin=time.perf_counter();owned=[]
        for i,shape in enumerate(SHAPES):
            assert outputs[i].buf and outputs[i].size==math.prod(shape)*4
            owned.append(np.ctypeslib.as_array(ct.cast(outputs[i].buf,ct.POINTER(ct.c_float)),shape=(math.prod(shape),)).copy().reshape(shape))
        timing['owned_copy']=time.perf_counter()-begin
    finally:
        if got or any(x.buf for x in outputs):
            begin=time.perf_counter();code=api.rknn_outputs_release(context,2,outputs)
            if code:errors.append('outputs_release returned '+str(code))
            else:counters['outputs_released']+=2
            base.check_call(code,'outputs_release');timing['outputs_release']=time.perf_counter()-begin
    begin=time.perf_counter();assert all(np.isfinite(y).all() for y in owned);timing['finite']=time.perf_counter()-begin
    timing['route_seconds']=time.perf_counter()-start
    return owned,timing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checked','compiled','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('checked-sha256','compiled-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--profile',action='store_true');p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--frequency-hz',type=int,choices=[1000000000],required=True);a=p.parse_args()
    assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes() and 1<=a.rounds<=10
    a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=False)
    pins={};contexts=[];api=base=None;affinity=os.sched_getaffinity(0)
    r=dict(schema='v78_original_pair_power2_smooth_NPU_probe',status='running',script_sha256=sha(__file__),
        original_model_only=True,candidate_accepted=False,realtime_validated=False,profile=a.profile,
        frequency_hz=a.frequency_hz,core_mask=7,init_flags=8 if a.profile else 0,
        API='All routes FP32 input pass_through=0 / want_float=1 owned FP32 outputs',
        timing_scope='Complete two-output SDK route incl conversion in inputs/outputs API, owned copy/release/finite; excludes hash, NPZ, profile and platform reads',
        schedule=list('ABC' if a.profile else 'ABCCBA'*a.rounds),calls=[],queries=[],cleanup_errors=[],contexts_created=0,contexts_destroyed=0,
        counters=dict(run_attempts=0,runs_succeeded=0,outputs_acquired=0,outputs_released=0))
    try:
        c=load(BASE/'src/deploy/rk3588/validation/original_pair_group/check_compile_pair_group.py',HELPER_SHA,'_pair_checks',pins)
        base=load(BASE/'src/deploy/rk3588/validation/run-v12-rknn-block-board.py',ABI_SHA,'_pair_abi',pins)
        def pin(path,digest):path=c.pin(path,digest);pins[str(path)]=digest;return path
        def artifact(root,row):path=c.artifact(root,row);pins[str(path)]=row['sha256'];return path
        base.require_board();os.sched_setaffinity(0,{4,5,6,7})
        import numpy as np
        parentpath=pin(BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json',PARENT_SHA)
        quantpath=pin(BASE/'experiments/vggt-v56-ptq-pairs-r1/compiled.json',QUANT_SHA)
        parent,quant=[json.loads(x.read_text()) for x in (parentpath,quantpath)]
        assert parent['blocks'][0]['sha256']==quant['blocks'][0]['sha256']==GRAPH_SHA
        checked=json.loads(pin(a.checked,a.checked_sha256).read_text());compiled=json.loads(pin(a.compiled,a.compiled_sha256).read_text())
        assert checked['status']=='checked_original_FP32_equivalence_not_quality_accepted' and checked['script_sha256']==CHECK_SOURCE_SHA
        assert checked['source_and_inputs_unchanged'] and not checked['cleanup_errors']
        assert compiled['status']=='compiled_not_quality_accepted' and compiled['script_sha256']==COMPILE_SOURCE_SHA
        assert compiled['checked_sha256']==a.checked_sha256 and compiled['check_source_sha256']==CHECK_SOURCE_SHA
        assert compiled['released'] and not compiled['cleanup_errors'] and compiled['source_and_inputs_unchanged'] and compiled['environment_restored']
        check=json.loads(pin(a.checked.parent/'check.json',checked['check_receipt_sha256']).read_text())
        assert compiled['check_receipt_sha256']==checked['check_receipt_sha256']
        assert check['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] and check['cases']==checked['cases']
        assert compiled['candidate_graph_sha256']==check['candidate_onnx_sha256']==checked['onnx']['sha256']
        artifact(a.checked.parent,checked['onnx'])
        models=[('A',artifact(parentpath.parent,parent['blocks'][0]['rknn'])),
                ('B',artifact(quantpath.parent,quant['blocks'][0]['rknn'])),('C',artifact(a.compiled.parent,compiled['rknn']))]
        cases=[]
        for case in checked['cases']:
            x=np.load(artifact(a.checked.parent,case['saved_input']),allow_pickle=False)
            assert x.dtype==np.float32 and list(x.shape)==SHAPES[0] and x.flags.c_contiguous and np.isfinite(x).all()
            refs=[np.load(artifact(a.checked.parent,item),allow_pickle=False) for item in case['baseline_outputs']]
            torchrefs=[np.load(artifact(Path(case['directory']),item),allow_pickle=False) for item in case['original_outputs']] if case['original_outputs'] else None
            cases.append((case,x,refs,torchrefs))
        assert len(cases)==2 and all(len(x[2])==2 for x in cases)
        r.update(cases=checked['cases'],checked_sha256=a.checked_sha256,compiled_sha256=a.compiled_sha256)
        pin(BASE/'src/deploy/rk3588/reference/rknn_api_v2.3.2.h',HEADER_SHA)
        api=base.bind_api(ct.CDLL(str(pin(Path('/usr/lib/librknnrt.so'),SDK_SHA))));routes={}
        for role,path in models:
            ctx=base.CONTEXT();contexts.append(ctx);name=ct.create_string_buffer(os.fsencode(path))
            code=api.rknn_init(ct.byref(ctx),ct.cast(name,ct.c_void_p),0,r['init_flags'],None)
            r['contexts_created']+=int(bool(ctx.value));base.check_call(code,'init');assert ctx.value
            base.check_call(api.rknn_set_core_mask(ctx,7),'core_mask')
            version=base.query(api,ctx,5,base.SDKVersion());assert re.match(r'^2\.3\.2(?:\D|$)',bytes(version.api_version).decode())
            counts=base.query(api,ctx,0,base.InputOutputNum());assert (counts.n_input,counts.n_output)==(1,2)
            descs=[io_desc(base,base.query(api,ctx,1,base.TensorAttr(index=0)),SHAPES[0],'input',0)]
            for i,shape in enumerate(SHAPES):descs.append(io_desc(base,base.query(api,ctx,2,base.TensorAttr(index=i)),shape,['frame_output','global_output'][i],i))
            r['queries'].append(dict(role=role,model_sha256=sha(path),descriptors=descs));routes[role]=ctx
        for case,x,refs,torchrefs in cases:
            first={}
            for sequence,role in enumerate(r['schedule']):
                before=sample();assert before['actual_frequency_hz']==a.frequency_hz and max(before['temperatures_c'].values())<75
                ys,timing=execute(base,api,np,routes[role],x,r['counters'],r['cleanup_errors'])
                after=sample();assert after['actual_frequency_hz']==a.frequency_hz and max(after['temperatures_c'].values())<75
                cold=role not in first
                if cold:first[role]=ys
                path=a.output_dir/f'{case["tag"]}-{sequence:02d}-{role}.npz';np.savez(path,output0=ys[0],output1=ys[1])
                row=dict(input=case['tag'],role=role,sequence=sequence,first_for_input_role=cold,component_seconds=timing['route_seconds'],timing=timing,
                    versus_original_ONNX=[c.compare(np,y,z) for y,z in zip(ys,refs)],
                    versus_original_Torch=[c.compare(np,y,z) for y,z in zip(ys,torchrefs)] if torchrefs else None,
                    repeat_first_role_byte_equal=[y.tobytes()==z.tobytes() for y,z in zip(ys,first[role])],
                    raw_sha256=[hashlib.sha256(y.tobytes()).hexdigest() for y in ys],
                    platform_before=before,platform_after=after,artifact=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)))
                if a.profile:row['profile']=base.perf_detail(api,routes[role])
                r['calls'].append(row)
        n=len(r['calls']);assert r['counters']==dict(run_attempts=n,runs_succeeded=n,outputs_acquired=2*n,outputs_released=2*n)
        if not a.profile:r['summary']={role:dict(warm_count=sum(x['role']==role and not x['first_for_input_role'] for x in r['calls']),
            mean_component_seconds=float(np.mean([x['component_seconds'] for x in r['calls'] if x['role']==role and not x['first_for_input_role']]))) for role in 'ABC'}
        r['status']='completed_component_diagnostic_not_quality_accepted'
    except BaseException as exc:r.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        for ctx in reversed(contexts):
            if ctx.value:
                try:base.check_call(api.rknn_destroy(ctx),'destroy');ctx.value=0;r['contexts_destroyed']+=1
                except BaseException as exc:r['cleanup_errors'].append(str(exc))
        r['live_contexts']=sum(bool(x.value) for x in contexts)
        try:os.sched_setaffinity(0,affinity)
        except BaseException as exc:r['cleanup_errors'].append('Affinity restore: '+str(exc))
        for path,digest in pins.items():
            try:assert sha(path)==digest,path
            except BaseException as exc:r['cleanup_errors'].append('Identity: '+str(exc))
        if r['cleanup_errors'] or r['live_contexts']:r['status']='failed_cleanup_or_identity'
        (a.output_dir/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:r.get(k) for k in ('status','summary','counters','contexts_created','contexts_destroyed','cleanup_errors','error')}),flush=True)
    return 0 if r['status'].startswith('completed_') else 1


if __name__=='__main__':raise SystemExit(main())
