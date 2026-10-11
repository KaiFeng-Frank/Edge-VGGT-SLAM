#!/usr/bin/env python3
"""Board-only, bounded half-IO diagnostic on the unchanged original pair00.

Three ABBA cycles, one normal flags=0/core7 context, no warmup/profiling. A is
FP32/pass0/want_float1; B freshly casts FP16/type1/pass0/want_float0. B copies
both logical FP16 outputs before release, then widens the owned copies. This
single-pair experiment cannot validate whole-model quality or real-time SLAM.
Optional pinned ARMv8.2 NEON conversion replaces only B's cast and widening.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
MANIFEST_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CHECKPOINT = dict(sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0', bytes=5026874952)
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
NEON_SOURCE_SHA = '77707b735a6291c417a997ed64acbe4e1cd60cc5f9dd150d83fe1836868bd5e3'
SHAPES = ((2,53,1024), (1,106,1024))
NAMES = ('frame_output', 'global_output')
sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):
            h.update(part)
    return h.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    if sha(path) != expected:
        raise ValueError('Required original artifact SHA mismatch: '+str(path))
    pins[str(path)] = expected
    return path


def artifact(directory, row, pins):
    path = (directory/row['file']).resolve(strict=True)
    if not path.is_relative_to(directory) or path.stat().st_size != row['bytes']:
        raise ValueError('Original artifact path/size differs')
    return pin(path,row['sha256'],pins)


def array(np, directory, item, shape, pins):
    value = np.load(artifact(directory,item,pins),allow_pickle=False)
    if (value.shape != shape or value.dtype != np.float32 or not value.flags.c_contiguous
            or not np.isfinite(value).all() or item['shape'] != list(shape) or item['dtype'] != 'float32'
            or value.nbytes != item['tensor_bytes'] or hashlib.sha256(value.tobytes()).hexdigest() != item['raw_sha256']):
        raise ValueError('Fixed original FP32 capture tensor differs')
    return value


def compare(np, a, b):
    if a.shape != b.shape or a.dtype != np.float32 or b.dtype != np.float32 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite original FP32 comparison shapes/dtypes required')
    delta = a.astype(np.float64)-b.astype(np.float64)
    norm, error = float(np.linalg.norm(b.astype(np.float64))),float(np.linalg.norm(delta))
    failed = int(np.count_nonzero(np.abs(delta)>1e-5+1e-4*np.abs(b.astype(np.float64))))
    return dict(finite=True,byte_equal=a.tobytes()==b.tobytes(),strict_rtol=1e-4,strict_atol=1e-5,
        strict_pass=failed==0,failed_element_count=failed,max_abs=float(np.abs(delta).max()),
        relative_l2=error/norm if norm else (0. if error==0 else None))


def native_convert(np, neon, source, dtype):
    if not source.flags.c_contiguous or source.dtype not in (np.dtype('float32'),np.dtype('float16')):
        raise ValueError('Native conversion requires dense FP32/FP16 storage')
    output=np.empty(source.shape,dtype=dtype)
    if np.shares_memory(source,output):
        raise ValueError('Native conversion requires disjoint source and destination buffers')
    if source.dtype==np.float32 and output.dtype==np.float16:
        neon.f32_to_f16(source.ctypes.data_as(ct.POINTER(ct.c_float)),
            output.ctypes.data_as(ct.POINTER(ct.c_uint16)),source.size)
    elif source.dtype==np.float16 and output.dtype==np.float32:
        neon.f16_to_f32(source.ctypes.data_as(ct.POINTER(ct.c_uint16)),
            output.ctypes.data_as(ct.POINTER(ct.c_float)),source.size)
    else:
        raise ValueError('Native conversion must change FP32 to FP16 or FP16 to FP32')
    return output


def execute(np, abi, api, context, source, mode, row, errors, neon=None):
    begin = time.perf_counter()
    tick = time.perf_counter()
    feed = (native_convert(np,neon,source,np.float16) if neon is not None else source.astype(np.float16)) if mode=='B' else source
    row['timing']['native_cast_seconds' if neon is not None else 'numpy_cast_seconds'] = time.perf_counter()-tick
    row['conversion_route']='native_NEON_FP16' if mode=='B' and neon is not None else ('NumPy_astype_FP16' if mode=='B' else 'original_FP32_no_conversion')
    if not np.isfinite(feed).all():
        raise ValueError('Explicit FP16 input cast is nonfinite')
    half = mode=='B'
    row.update(input_api_type=1 if half else 0,input_pass_through=0,want_float=0 if half else 1,
        input_bytes=feed.nbytes,input_raw_sha256=hashlib.sha256(feed.tobytes()).hexdigest())
    inputs = (abi.Input*1)(abi.Input(0,feed.ctypes.data,feed.nbytes,0,1 if half else 0,3))
    outputs = (abi.Output*2)()
    for i in range(2): outputs[i].index,outputs[i].want_float=i,0 if half else 1
    run,received = abi.RunExtend(0,0,30000,-1),abi.OutputExtend()
    got,owned = False,[]
    def call(name, function):
        tick=time.perf_counter(); code=function(); elapsed=time.perf_counter()-tick
        row['api_calls'].append(dict(operation=name,code=int(code)))
        row['timing'][name+'_seconds']=elapsed
        if name=='rknn_outputs_release' and code: errors.append(f'{name} returned {code}')
        abi.check_call(code,name)
    try:
        call('rknn_inputs_set',lambda:api.rknn_inputs_set(context,1,inputs))
        call('rknn_run',lambda:api.rknn_run(context,ct.byref(run)))
        call('rknn_outputs_get',lambda:api.rknn_outputs_get(context,2,outputs,ct.byref(received)))
        got=True
        if run.frame_id != received.frame_id: raise RuntimeError('Synchronous pair frame ID differs')
        row.update(runtime_frame_id=int(run.frame_id),output_bytes=[int(o.size) for o in outputs])
        tick=time.perf_counter()
        for i,shape in enumerate(SHAPES):
            if not outputs[i].buf or outputs[i].size != math.prod(shape)*(2 if half else 4):
                raise ValueError('SDK logical output pointer/type/size differs')
            pointer=ct.cast(outputs[i].buf,ct.POINTER(ct.c_uint16 if half else ct.c_float))
            view=np.ctypeslib.as_array(pointer,shape=(math.prod(shape),))
            if half: view=view.view(np.float16)
            owned.append(view.copy().reshape(shape))
        row['timing']['owned_copy_seconds']=time.perf_counter()-tick
    finally:
        try:
            if got or any(o.buf for o in outputs):
                call('rknn_outputs_release',lambda:api.rknn_outputs_release(context,2,outputs))
                row['outputs_released']=True
        finally:
            row['instrumented_sdk_route_seconds']=time.perf_counter()-begin
    tick=time.perf_counter()
    widened=[native_convert(np,neon,x,np.float32) if neon is not None else x.astype(np.float32) for x in owned] if half else owned
    row['timing']['widen_seconds']=time.perf_counter()-tick if half else 0.
    row['measured_stage_sum_seconds']=sum(row['timing'].values())
    row['instrumented_route_seconds']=time.perf_counter()-begin
    if not all(np.isfinite(x).all() for x in widened): raise ValueError('Nonfinite owned pair output')
    return widened,owned


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=BASE/'src')
    p.add_argument('--manifest',type=Path,default=BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--capture-dir',type=Path,default=BASE/'experiments/vggt-v53-capture-r1')
    p.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--neon-library',type=Path,help='Optional board-built ARMv8.2 conversion library; requires SHA256')
    p.add_argument('--neon-library-sha256',help='Exact SHA256 of the optional native conversion library')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); pins={}
    if (args.neon_library is None)!=(args.neon_library_sha256 is None):
        p.error('Supply --neon-library and --neon-library-sha256 together')
    if args.neon_library_sha256 is not None and (len(args.neon_library_sha256)!=64 or any(c not in '0123456789abcdef' for c in args.neon_library_sha256)):
        p.error('--neon-library-sha256 must be a lowercase 64-digit SHA256')
    abi_path=pin(args.source_root/'deploy/rk3588/validation/run-v12-rknn-block-board.py',ABI_SHA,pins)
    spec=importlib.util.spec_from_file_location('_v56_original_half_io_ABI',abi_path)
    abi=importlib.util.module_from_spec(spec); spec.loader.exec_module(abi)
    board=abi.require_board()  # Before NumPy imports, library loading or context creation.
    directory=args.output_dir.resolve(); directory.mkdir(parents=True,exist_ok=False)
    report=dict(status='failed',scope='original VGGT pair00 SDK IO diagnostic only',board=board,
        script_sha256=sha(__file__),init_flags=0,core_mask=7,profile=False,warmup_calls=0,
        order=list('ABBA'*3),calls=[],api_calls=[],cleanup_errors=[],candidate_accepted=False,
        quality_accepted=False,real_time_validated=False,model_inference_on_host=False,
        half_conversion_route='native_NEON_FP16' if args.neon_library is not None else 'NumPy_astype_FP16',
        timing_scope='separate measured stage sum; instrumented route also includes guards/hash/ABI preparation')
    started=time.monotonic(); context,api=abi.CONTEXT(),None; affinity=os.sched_getaffinity(0)
    env_keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    original_env={k:os.environ.get(k) for k in env_keys}
    try:
        if 4 not in affinity: raise RuntimeError('A76 CPU4 is unavailable')
        os.sched_setaffinity(0,{4}); report['cpu_affinity']=[4]
        for k in env_keys: os.environ[k]='1'
        import numpy as np
        report['numpy_version']=np.__version__
        pin(args.source_root/'deploy/rk3588/reference/rknn_api_v2.3.2.h',HEADER_SHA,pins)
        library=pin(args.library,LIBRARY_SHA,pins); pins[str(Path(__file__).resolve())]=report['script_sha256']
        neon=None
        if args.neon_library is not None:
            neon_source=pin(Path(__file__).with_name('fp16_neon.c'),NEON_SOURCE_SHA,pins)
            neon_path=pin(args.neon_library,args.neon_library_sha256,pins)
            report['native_conversion']=dict(source_file=str(neon_source),source_sha256=NEON_SOURCE_SHA,
                library_file=str(neon_path),library_sha256=args.neon_library_sha256,
                storage_types=['float32','uint16_t_FP16_bits'],disjoint_buffers_enforced=True,
                vector_alignment_required=False,FPCR_policy='unchanged; requires round-to-nearest and no flush-to-zero')
            neon=ct.CDLL(str(neon_path))
            neon.f32_to_f16.argtypes=[ct.POINTER(ct.c_float),ct.POINTER(ct.c_uint16),ct.c_size_t]
            neon.f16_to_f32.argtypes=[ct.POINTER(ct.c_uint16),ct.POINTER(ct.c_float),ct.c_size_t]
            neon.f32_to_f16.restype=neon.f16_to_f32.restype=None
        manifest_path=pin(args.manifest,MANIFEST_SHA,pins); manifest=json.loads(manifest_path.read_text())
        capture_path=pin(args.capture_dir/'capture.json',CAPTURE_SHA,pins); capture=json.loads(capture_path.read_text())
        if (manifest['schema']!='v55_original_VGGT_24_frame_global_pair_RKNN_manifest'
                or manifest['capture_sha256']!=CAPTURE_SHA or manifest['checkpoint']!=CHECKPOINT
                or capture['schema']!='v53_original_VGGT_48_block_capture' or capture['checkpoint']!=CHECKPOINT
                or capture['status']!='completed_original_FP32_full_forward_and_48_block_capture' or capture['cleanup_errors']):
            raise ValueError('Pinned unchanged original pair/capture required')
        pair=manifest['blocks'][0]
        if pair['role']!='pair' or pair['index']!=0 or pair['input_shape']!=list(SHAPES[0]):
            raise ValueError('Only frozen original pair00 is supported')
        model=artifact(manifest_path.parent,pair['rknn'],pins)
        blocks={r['role']:r for r in capture['blocks'] if r['index']==0}
        source=array(np,args.capture_dir,blocks['frame']['input'],SHAPES[0],pins)
        references=[array(np,args.capture_dir,blocks[role]['output'],shape,pins) for role,shape in zip(('frame','global'),SHAPES)]
        if blocks['global']['input']['raw_sha256']!=blocks['frame']['output']['raw_sha256']:
            raise ValueError('Original global reference must use original frame output')
        for role in ('frame','global'):
            if pair['parameter_raw_sha256'][role]!={k:v['raw_sha256'] for k,v in blocks[role]['parameters'].items()}:
                raise ValueError('Frozen original parent parameter identity differs')
        report.update(manifest_sha256=MANIFEST_SHA,capture_sha256=CAPTURE_SHA,model_sha256=pair['rknn']['sha256'],
            source_and_input_hashes=pins.copy(),memory_before_sdk=abi.process_memory())
        api=abi.bind_api(ct.CDLL(str(library)))
        def call(name,function):
            tick=time.perf_counter(); code=function()
            report['api_calls'].append(dict(operation=name,code=int(code),seconds=time.perf_counter()-tick))
            abi.check_call(code,name)
        filename=ct.create_string_buffer(str(model).encode())
        call('rknn_init',lambda:api.rknn_init(ct.byref(context),ct.cast(filename,ct.c_void_p),0,0,None))
        if not context.value: raise RuntimeError('RKNN returned an empty context')
        call('rknn_set_core_mask',lambda:api.rknn_set_core_mask(context,7))
        def query(command,value):
            call(f'rknn_query({command},index={getattr(value,"index",0)})',lambda:api.rknn_query(context,command,ct.byref(value),ct.sizeof(value)))
            return value
        version=query(5,abi.SDKVersion()); report['sdk_version']=bytes(version.api_version).decode()
        if not report['sdk_version'].startswith('2.3.2'): raise ValueError('RKNN runtime 2.3.2 required')
        count=query(0,abi.InputOutputNum())
        if (count.n_input,count.n_output)!=(1,2): raise ValueError('One input and both original outputs required')
        attrs=[abi.attr_record(query(cmd,abi.TensorAttr(index=index))) for cmd,index in ((1,0),(2,0),(2,1))]
        report['logical_attrs']=attrs
        for a,index,name,shape in zip(attrs,(0,0,1),('input',*NAMES),(SHAPES[0],*SHAPES)):
            identity=a['quantization_type']==2 and a['scale']==1. and a['zero_point']==0
            if (a['index']!=index or a['name']!=name or a['shape']!=list(shape) or a['format']!=3
                    or a['type']!=1 or a['n_elems']!=math.prod(shape) or a['size']!=2*math.prod(shape)
                    or a['pass_through']!=0 or (a['quantization_type']!=0 and not identity) or not math.isfinite(a['scale'])):
                raise ValueError('Dense logical FP16 query shape/fmt/type/size/name/index differs')
        report['memory_after_init']=abi.process_memory(); first_A=None
        for index,mode in enumerate(report['order']):
            row=dict(index=index,cycle=index//4,mode=mode,phase='first' if index==0 else 'subsequent',
                first_for_mode=index in (0,1),timing={},api_calls=[])
            report['calls'].append(row)
            values,raw=execute(np,abi,api,context,source,mode,row,report['cleanup_errors'],neon=neon)
            if first_A is None: first_A=values
            row['versus_original_FP32']={name:compare(np,a,b) for name,a,b in zip(NAMES,values,references)}
            row['versus_first_A']={name:compare(np,a,b) for name,a,b in zip(NAMES,values,first_A)}
            saved=directory/f'call-{index:02d}-{mode}.npz'
            np.savez(saved,**dict(zip(NAMES,values)),**({f'{n}_raw_FP16':v for n,v in zip(NAMES,raw)} if mode=='B' else {}))
            row.update(status='completed',output=dict(file=saved.name,bytes=saved.stat().st_size,sha256=sha(saved)))
            print(json.dumps(dict(index=index,mode=mode,stages=row['timing'],versus_first_A=row['versus_first_A']),allow_nan=False),flush=True)
        report['timing_summary']={mode:{k:float(np.mean([r['timing'][k] for r in report['calls'] if r['mode']==mode]))
            for k in report['calls'][0]['timing']} for mode in ('A','B')}
        report['status']='completed_diagnostic_not_accuracy_acceptance'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if api is not None and context.value:
            try:
                code=api.rknn_destroy(context); report['api_calls'].append(dict(operation='rknn_destroy',code=int(code)))
                report['context_destroyed']=code==0
                if code: report['cleanup_errors'].append(f'rknn_destroy returned {code}')
            except BaseException as exc:
                report['context_destroyed']=False
                report['cleanup_errors'].append(f'Context destroy: {exc}')
            finally: context.value=0
        try:
            report['source_and_input_hashes_after']={p:sha(p) for p in pins}
            report['source_and_inputs_unchanged']=report['source_and_input_hashes_after']==pins
            if not report['source_and_inputs_unchanged']: report['cleanup_errors'].append('Pinned source/input changed')
        except BaseException as exc: report['cleanup_errors'].append(f'Pin recheck: {exc}')
        for k,v in original_env.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v
        try:
            os.sched_setaffinity(0,affinity); report['affinity_restored']=os.sched_getaffinity(0)==affinity
        except OSError as exc: report['cleanup_errors'].append(f'Affinity restore: {exc}')
        report['memory_final']=abi.process_memory(); report['total_seconds']=time.monotonic()-started
        report['completed_calls']=sum(r.get('status')=='completed' for r in report['calls'])
        if report['cleanup_errors']: report['status']='failed'
        with (directory/'result.json').open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    return 0 if report['status']=='completed_diagnostic_not_accuracy_acceptance' else 1


if __name__=='__main__':
    raise SystemExit(main())
