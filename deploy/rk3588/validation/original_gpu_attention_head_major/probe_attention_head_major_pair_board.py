#!/usr/bin/env python3
"""Board-only matched original frame0/global0 pair: compiled NPU vs GPU tails.

Every timed call includes the full original pair, FP16 I/O conversion, SDK
custom callbacks, owned output copies and finite checks. No hidden warmup.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
HELPER_SHA = 'f3801ba5b92c1043aab2d3ebc6c0ad3258b8633c25e216b1298fbb1e0c9c38a0'
CAPTURE_SHA = '3b4eb0b24d320699dd7d198fd3158454a3043ed7244a0290b267cbfc450e8fee'
EXPORT_SHA = 'e563cfcef6934da57e1f9a5c63cc6ded346661bc81e9966160ed3de78f200402'
COMPILE_SHA = '09c6342dc31c8ece8a7f8640c7aa8bbde8cbaa175edb0e1f8502c42ffc684f38'
COMPILER_SHA = 'f6a480ba36f2291ed42ed99c9b575203f7b0d693f53a9b751878b0fcceff80c2'
BUILDER_SHA = 'deaa04cccf731c98db987bec26e7d69bd73c29d46d1e5f463c997208c379b225'
LEGACY_SHA = 'f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2'
MALI_SHA = 'd57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda'
COMPONENT_SHA = '9dc9e66b9b0521ff168e31063c47bd21c6dc4cbccc9bd339a4ecf28b647b33e7'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compiled', 'runtime-source', 'borrowed-source', 'abi', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('compiled', 'runtime-source', 'borrowed-source', 'abi'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--rounds', type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pins, contexts, handle = {}, [], None
    api = base = None
    affinity = os.sched_getaffinity(0)
    report = dict(schema='v84_original_pair_head_major_custom_GPU_attention_AB', status='failed',
        original_model_only=True, candidate_accepted=False, full_model_or_realtime_validated=False,
        script_sha256=sha(__file__), schedule=list('ABBA' * args.rounds),
        normal_init_flags=0, NPU_core_mask=7, NPU_expected_hz=1000000000,
        GPU_governor_unchanged=True, frequency_written=False, hidden_warmup_calls=0,
        custom_output_layout='NCHW original [B,16,S,64]', original_consumer_layout_nodes_retained=True,
        first_exclusion='First call for each input and route excluded from warm statistics',
        timing_scope='Complete original pair with NEON FP16 I/O, one SDK run including custom GPU callbacks, owned outputs, SDK release and finite checks; platform/hash/save/compare excluded',
        routes={'A':'original_compiled_FP16_pair', 'B':'same_pair_with_two_original_FP32_GPU_head_major_tails'},
        calls=[], queries=[], contexts_created=0, contexts_destroyed=0, cleanup_errors=[],
        counters=dict(sdk_run_attempts=0,sdk_runs_succeeded=0,outputs_acquired=0,outputs_released=0))
    def pin(path, digest):
        path = Path(path).resolve(strict=True)
        require(sha(path) == digest, 'Frozen source/artifact changed: ' + str(path))
        pins[path] = digest
        return path
    def load(path, digest, name):
        path = pin(path, digest)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    def artifact(root, row):
        root = Path(root).resolve(strict=True)
        path = (root / row['file']).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == row['bytes'], 'Artifact path/size changed')
        return pin(path, row['sha256'])
    def gpu_state():
        root = Path('/sys/class/devfreq/fb000000.gpu')
        return {key: (root/key).read_text().strip() for key in ('governor','min_freq','max_freq')}
    try:
        root = BASE/'src'
        helper = load(root/'deploy/rk3588/validation/original_pair_group/probe_original_pair_group.py', HELPER_SHA, '_v83_pair_helper')
        base = load(root/'deploy/rk3588/validation/run-v12-rknn-block-board.py', helper.ABI_SHA, '_v83_pair_abi')
        report['board'] = base.require_board()
        require({4,5,6,7} <= affinity, 'Big CPU cores unavailable')
        os.sched_setaffinity(0, {4,5,6,7})
        import numpy as np
        parent_path = pin(BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json', helper.PARENT_SHA)
        parent = json.loads(parent_path.read_text())
        original = parent['blocks'][0]
        require(original['role']=='pair' and original['index']==0, 'Original complete pair00 required')
        require(args.compiled_sha256 == COMPILE_SHA, 'Only the frozen v84 compile receipt is allowed')
        compiled_path = pin(args.compiled, args.compiled_sha256)
        compiled = json.loads(compiled_path.read_text())
        require(compiled['schema']=='v84_original_pair_attention_head_major_custom_FP16_compile'
                and compiled['script_sha256']==COMPILER_SHA and compiled['builder_source_sha256']==BUILDER_SHA
                and compiled['original_parent_manifest_sha256']==helper.PARENT_SHA
                and compiled['original_pair_onnx_sha256']=='efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
                and compiled['status']=='compiled_custom_callbacks_not_runtime_validated'
                and compiled['released'] and compiled['source_and_inputs_unchanged']
                and not compiled['cleanup_errors'] and compiled['export_manifest_sha256']==EXPORT_SHA,
                'Completed original custom compile required')
        export_path = pin(BASE/'experiments/vggt-v84-head-major-export-r1/export.json', EXPORT_SHA)
        capture_path = pin(BASE/'experiments/vggt-v83-attention-tail-capture-r1/capture.json', CAPTURE_SHA)
        capture = json.loads(capture_path.read_text())
        require(capture['status']=='completed_original_FP32_attention_tail_capture' and not capture['cleanup_errors'], 'Original capture required')
        runtime = load(args.runtime_source,args.runtime_source_sha256,'_v83_custom_runtime')
        borrowed = load(args.borrowed_source,args.borrowed_source_sha256,'_v83_borrowed_bridge')
        component = load(BASE/'experiments/vggt-v83-source-r1/original_attention_tail_opencl.py',COMPONENT_SHA,'_v84_original_tail_component')
        legacy = load(root/'deploy/rk3588/validation/original_native_gpu/native_erf_gelu_opencl.py',LEGACY_SHA,'_v83_custom_legacy')
        pin(args.abi,args.abi_sha256)
        library = pin(BASE/'experiments/libmali-valhall-g610-g6p0-dummy.so',MALI_SHA)
        api = base.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so',helper.SDK_SHA))))
        neon = ct.CDLL(str(pin(BASE/'experiments/vggt-v56-neon-build-r1/fp16_neon.so',helper.NEON_SHA)))
        for name in ('f32_to_f16','f16_to_f32'):
            fn = getattr(neon,name)
            fn.argtypes, fn.restype = [ct.c_void_p,ct.c_void_p,ct.c_size_t],None
        expected = original['output_bindings']
        require([x['shape'] for x in expected]==[[2,53,1024],[1,106,1024]], 'Original pair outputs required')
        models = [('A',artifact(parent_path.parent,original['rknn'])),('B',artifact(compiled_path.parent,compiled['rknn']))]
        routes = {}
        report['GPU_baseline_governor_limits'] = gpu_state()
        for role, model in models:
            context=base.CONTEXT()
            contexts.append(context)
            filename=ct.create_string_buffer(os.fsencode(model))
            code=api.rknn_init(ct.byref(context),ct.cast(filename,ct.c_void_p),0,0,None)
            report['contexts_created'] += int(bool(context.value))
            base.check_call(code,'rknn_init '+role)
            require(bool(context.value),'Empty context')
            if role=='B':
                handle=runtime.register_original_attention_custom(api,context,base,export_path,EXPORT_SHA,
                    borrowed_module=borrowed,legacy_module=legacy,component_module=component,opencl_library=str(library),
                    abi_path=args.abi,abi_sha256=args.abi_sha256)
                report['custom_after_registration']=handle.metrics()
            base.check_call(api.rknn_set_core_mask(context,7),'rknn_set_core_mask')
            counts=base.query(api,context,0,base.InputOutputNum())
            require((counts.n_input,counts.n_output)==(1,2),'Original one-input/two-output ABI required')
            descriptors=[]
            for command,bindings in ((1,[dict(index=0,name='input',shape=[2,53,1024])]),(2,expected)):
                for binding in bindings:
                    attr=base.query(api,context,command,base.TensorAttr(index=binding['index']))
                    desc=base.attr_record(attr)
                    require(desc['shape']==binding['shape'] and desc['index']==binding['index']
                        and desc['name']==binding['name'] and desc['type']==1 and desc['format']==3
                        and desc['n_elems']==math.prod(binding['shape']) and desc['size']==2*math.prod(binding['shape'])
                        and desc['pass_through']==0,'Dense FP16 original pair ABI changed: '+str(desc))
                    descriptors.append(desc)
            report['queries'].append(dict(route=role,model_sha256=sha(model),descriptors=descriptors))
            routes[role]=context
        for case in capture['cases']:
            data_root=BASE/('experiments/vggt-v53-capture-r1' if case['tag']=='v53' else 'experiments/vggt-v56-calibration-r1')
            feed=np.load(artifact(data_root,case['original_frame_input']),allow_pickle=False)
            original_hash=hashlib.sha256(feed.tobytes()).hexdigest()
            require(original_hash==case['original_frame_input']['raw_sha256'],'Original pair input raw hash differs')
            refs=[np.load(artifact(capture_path.parent,case['roles'][role]['artifacts']['block_output']),allow_pickle=False) for role in ('frame','global')]
            first={}
            for sequence,role in enumerate(report['schedule']):
                before=helper.sample()
                require(before['actual_frequency_hz']==1000000000 and max(before['temperatures_c'].values())<75,'NPU clock/temperature out of bound')
                values,timing=helper.execute(base,api,neon,np,routes[role],feed,expected,report['counters'],report['cleanup_errors'])
                after=helper.sample()
                require(after['actual_frequency_hz']==1000000000 and max(after['temperatures_c'].values())<75,'NPU clock/temperature out of bound')
                require(gpu_state()==report['GPU_baseline_governor_limits'],'GPU governor/limits changed')
                cold=role not in first
                if cold:first[role]=values
                path=args.output_dir/f'{case["tag"]}-{sequence:02d}-{role}.npz'
                np.savez(path,frame=values[0],global_output=values[1])
                report['calls'].append(dict(case=case['tag'],sequence=sequence,route=role,first=cold,timing=timing,
                    before=before,after=after,GPU_observed_hz=int(Path('/sys/class/devfreq/fb000000.gpu/cur_freq').read_text()),
                    original_CPU=[helper.compare(np,x,y) for x,y in zip(values,refs)],
                    same_input_first_A=[helper.compare(np,x,y) for x,y in zip(values,first['A'])],
                    repeat_first_route=[helper.compare(np,x,y) for x,y in zip(values,first[role])],
                    saved_output=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))))
            require(hashlib.sha256(feed.tobytes()).hexdigest()==original_hash,'Input mutated')
        report['custom_after_calls']=handle.metrics()
        runs=len(capture['cases'])*len(report['schedule'])
        require(report['counters']==dict(sdk_run_attempts=runs,sdk_runs_succeeded=runs,
                outputs_acquired=2*runs,outputs_released=2*runs),'Whole pair SDK/output lifetime invariant failed')
        report['summary']={role:dict(warm_calls=len(rows),mean_seconds=float(np.mean(rows)),p95_seconds=float(np.percentile(rows,95)))
            for role in 'AB' for rows in [[c['timing']['route_seconds'] for c in report['calls'] if c['route']==role and not c['first']]]}
        report['status']='completed_component_not_quality_accepted'
    except BaseException as exc:
        if handle is None:
            handle=getattr(exc,'custom_handle',None)
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        safe=True
        if handle is not None:
            try:
                handle.close()
                report['custom_after_close']=handle.metrics()
            except BaseException as exc:
                safe=False
                report['cleanup_errors'].append('GPU close: '+str(exc))
        for context in reversed(contexts):
            if context.value and safe:
                try:
                    base.check_call(api.rknn_destroy(context),'rknn_destroy')
                    context.value=0
                    report['contexts_destroyed']+=1
                except BaseException as exc:report['cleanup_errors'].append('Context close: '+str(exc))
        if handle is not None:report['custom_after_context_destroy']=handle.metrics()
        report['live_contexts']=sum(bool(c.value) for c in contexts)
        report['source_and_inputs_unchanged']=all(sha(path)==digest for path,digest in pins.items())
        report['source_and_artifact_sha256']={str(path):digest for path,digest in pins.items()}
        try:
            os.sched_setaffinity(0,affinity)
            report['affinity_restored']=os.sched_getaffinity(0)==affinity
        except BaseException as exc:
            report['affinity_restored']=False
            report['cleanup_errors'].append('Affinity restore: '+str(exc))
        if report['cleanup_errors'] or report['live_contexts'] or not report['source_and_inputs_unchanged'] or not report['affinity_restored']:
            report['status']='failed_cleanup_or_identity'
        (args.output_dir/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:report.get(k) for k in ('status','error','summary','cleanup_errors')}),flush=True)
    return 0 if report['status']=='completed_component_not_quality_accepted' else 1

if __name__=='__main__':
    raise SystemExit(main())
