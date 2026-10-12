#!/usr/bin/env python3
"""Original five-pair group0 versus same-weight FC2 K4, ordinary FP16 I/O.

Two real saved inputs, ABBA*6 each, two persistent contexts. Component-only;
the candidate changes FP32 reduction order and is not a quality acceptance.
The published original retained runtime owns all SDK packing/copy/release.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

BASE = Path('/mnt/vggt-data/vggt-slam2')
PROFILE_HELPER_SHA = '6a7c095eea408b07cf227561a24e69a773f405945c813cf83265b5f63744b92f'
BUILD_SHA = '7697ac2c7c561f48f5a0289ef5808cf6b26d4bc2d7525ae4b9b911296241709b'
CHECK_SHA = 'd970e42d437ad7fa05e94c24126e41d303ba28f7961565b382022370f820d3fb'
CHECK_RESULT_SHA = '3501e82501f1ca6f3e10cb53399b227d722c371b02810ae4faa8d4c8a1828794'
ORDER = 'left_associative_in_increasing_K_chunk_order'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def comparison(np, actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    difference = np.abs(a - b); outside = difference > 1e-5 + 1e-4 * np.abs(b)
    norm = float(np.linalg.norm(b.reshape(-1))); error = float(np.linalg.norm((a-b).reshape(-1)))
    return dict(strict_rtol=1e-4, strict_atol=1e-5, strict_pass=not bool(outside.any()),
        outside_tolerance=int(outside.sum()), elements=int(actual.size), max_abs=float(difference.max()),
        relative_l2=error/norm if norm else (0. if error == 0 else None),
        byte_equal=actual.tobytes() == reference.tobytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=BASE/'src')
    parser.add_argument('--profile-helper', type=Path, default=Path(__file__).with_name('probe_retained_profiles_board.py'))
    parser.add_argument('--manifest', type=Path, default=BASE/'experiments/vggt-v67-retained-groups-r1/compiled.json')
    parser.add_argument('--candidate-result', type=Path, default=BASE/'experiments/vggt-v76-fc2-k4-group0-compile-r1/result.json')
    parser.add_argument('--candidate-result-sha256', required=True)
    parser.add_argument('--expected-npu-hz', type=int, choices=(300000000,1000000000), default=1000000000)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=False)
    old_affinity = os.sched_getaffinity(0)
    pins, contexts, api, abi = {}, [], None, None
    report = dict(schema='v76_original_retained_group0_FC2_K4_SDK_comparison', status='failed',
        script_sha256=sha(__file__), candidate_result_sha256=args.candidate_result_sha256,
        original_model_only=True, actual_full_Torch_model_OR_ORT_or_compiler_execution=False,
        actual_SDK_component_inference=True, group_index=0, indices=list(range(5)),
        FP32_reduction_order_changed=True, partial_sum_order=ORDER,
        candidate_accepted=False, quality_accepted=False, full_model_or_realtime_validated=False,
        init_flags=0, core_mask=7, timeout_ms=30000, hidden_warmup_calls=0,
        schedule_per_input='ABBA repeated6; two cases in fixed original order',
        warm_scope='Exclude first call of each route within each input:11 warm/case/route,22 total/route; second-case first is not a fresh-context cold call',
        timing_scope='Published original _run_group wall time including NEON conversions, SDK calls, owned copy/release/widen/finite; excludes platform reads, comparison/hash and NPZ saves',
        frequency_written=False, expected_npu_hz=args.expected_npu_hz,
        frequency_scope='Before and after each call samples only, not proof of continuous frequency',
        contexts_created=0, contexts_destroyed=0, cleanup_errors=[], calls=[], queries=[])
    stats = dict(sdk_run_attempts=0, sdk_runs_succeeded=0, outputs_acquired=0, outputs_released=0,
        sdk_io_seconds=0., cleanup_errors=report['cleanup_errors'],
        half_io=dict(calls_attempted=0, calls_completed=0, input_conversion_calls=0, output_conversion_calls=0,
            **{key+'_seconds':0. for key in ('input_conversion','output_conversion','owned_half_copy',
                'sdk_api','route','rknn_inputs_set','rknn_run','rknn_outputs_get','rknn_outputs_release')}))
    def pin(path, digest):
        path=Path(path).resolve(strict=True); require(sha(path)==digest,'Artifact SHA differs: '+str(path));pins[path]=digest;return path
    def artifact(directory, item):
        directory=Path(directory).resolve(strict=True);relative=Path(item['file'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Relative artifact required')
        path=(directory/relative).resolve(strict=True)
        require(path.is_relative_to(directory) and path.is_file() and path.stat().st_size==item['bytes'],'Artifact path/size differs')
        return pin(path,item['sha256'])
    def load(path,digest,name):
        spec=importlib.util.spec_from_file_location(name,pin(path,digest));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    def destroy(context):
        if context.value:
            try:
                abi.check_call(api.rknn_destroy(context),'rknn_destroy');context.value=0;report['contexts_destroyed']+=1
            except BaseException as exc:report['cleanup_errors'].append('Destroy: '+str(exc))
    try:
        helper=load(args.profile_helper,PROFILE_HELPER_SHA,'_v76_group_profile_helpers')
        abi=load(args.repo/'deploy/rk3588/validation/run-v12-rknn-block-board.py',helper.ABI_SHA,'_v76_K4_ABI')
        report['board']=abi.require_board()
        pin(args.repo/'vggt_slam/__init__.py',hashlib.sha256(b'').hexdigest())
        pin(args.repo/'vggt_slam/rknn_aggregator_blocks.py',helper.HELPER_SHA)
        pin(args.repo/'deploy/rk3588/reference/rknn_api_v2.3.2.h',helper.HEADER_SHA)
        pin(args.repo/'vggt_slam/native/fp16_neon.c',helper.NEON_SOURCE_SHA)
        sys.path.insert(0,str(args.repo.resolve(strict=True)))
        rt=load(args.repo/'vggt_slam/rknn_retained_aggregator.py',helper.RUNTIME_SHA,'_v76_K4_retained_route')
        require(Path(rt.original.__file__).resolve()==(args.repo/'vggt_slam/rknn_aggregator_blocks.py').resolve(),'Helper import escaped repo')
        require('torch' not in sys.modules,'This component comparison must not load Torch')
        import numpy as np
        manifest_path=pin(args.manifest,helper.MANIFEST_SHA);manifest=json.loads(manifest_path.read_text());directory=manifest_path.parent
        require(manifest['schema']=='v67_original_VGGT_retained_groups_RKNN_manifest' and len(manifest['blocks'])==4
            and manifest['status']=='compiled_not_quality_accepted' and manifest['cleanup_errors']==[]
            and manifest['original_parent_manifest_sha256']==rt.PARENT_SHA and manifest['original_target_tokens_global_index']==20,'Original complete group manifest differs')
        row=manifest['blocks'][0];bindings=rt.bindings(0)
        require(row['index']==0 and row['indices']==list(range(5)) and row['output_bindings']==bindings,'Only original group0 two outputs supported')
        parent=json.loads(pin(Path(manifest['original_parent_manifest_directory'])/'compiled.json',rt.PARENT_SHA).read_text())
        require(row['parents']==parent['blocks'][:5],'Original five-pair lineage differs')
        for relative,digest in manifest['source_hashes'].items():
            path=(args.repo/relative).resolve(strict=True);require(path.is_relative_to(args.repo.resolve()),'Source escaped repo');pin(path,digest)
        artifact(directory,row);original_model=artifact(directory,row['rknn'])
        original_check=json.loads(artifact(directory,row['check_receipt']).read_text())
        rt.RKNNRetainedOriginalAggregator._check_receipts(None,directory,row,0,manifest['export_manifest_sha256'],original_model)
        candidate_path=pin(args.candidate_result,args.candidate_result_sha256);candidate_dir=candidate_path.parent
        compiled=json.loads(candidate_path.read_text());checked_path=pin(candidate_dir/'check.json',CHECK_SHA);checked=json.loads(checked_path.read_text())
        require(compiled['schema']=='v73_original_VGGT_retained_FC2_K4_build' and compiled['status']=='compiled_not_quality_accepted'
            and compiled['mode']=='compile_checked' and compiled['script_sha256']==BUILD_SHA
            and compiled['helper_sha256']=='21d958ef385eaa419e81bef04303f07ffb5b7bb32a4854cd5d17ffba6247680b'
            and compiled['baseline_group']==row and compiled['original_pair_parents']==row['parents']
            and compiled['baseline_manifest_sha256']==helper.MANIFEST_SHA and compiled['original_parent_manifest_sha256']==rt.PARENT_SHA
            and compiled['group_index']==0 and compiled['indices']==list(range(5)) and compiled['chunks']==4
            and compiled['baseline_graph_sha256']==row['sha256'] and compiled['output_bindings']==bindings
            and compiled['public_output_sources']==row['public_output_sources'] and compiled['check_receipt_sha256']==CHECK_SHA
            and compiled['check_result_sha256']==CHECK_RESULT_SHA and compiled['configuration']==rt.CONFIG and compiled['build_options']==rt.BUILD
            and compiled['released'] is True and compiled['release_code'] in (None,0) and compiled['cleanup_errors']==[]
            and compiled['source_and_inputs_unchanged'] is True and compiled['environment_restored'] is True
            and compiled['candidate_accepted'] is False and compiled['quality_accepted'] is False
            and compiled['FP32_reduction_order_changed'] is True and compiled['partial_sum_order']==ORDER
            and [o['name'] for o in compiled['operations']]==['config','load_onnx','build','export_rknn']
            and all(o['status']=='completed' and o['code'] in (None,0) for o in compiled['operations']), 'Successful pinned same-source K4 compile required')
        require(compiled['source_and_input_hashes']==compiled['source_and_input_hashes_after'],'Compile source/input hashes differ')
        source_paths=[Path(path) for path,digest in compiled['source_and_input_hashes'].items() if digest==BUILD_SHA]
        require(len(source_paths)==1,'Exactly one frozen candidate compiler source required');pin(source_paths[0],BUILD_SHA)
        proof=compiled['graph_proof'];required_flags=('all_other_original_nodes_protobuf_unchanged','all_other_original_initializers_protobuf_unchanged',
            'original_FC1_nodes_protobuf_unchanged','original_GELU_nodes_protobuf_unchanged','original_FC2_bias_Add_nodes_protobuf_unchanged',
            'original_LayerScale_and_residual_nodes_protobuf_unchanged','original_IO_protobuf_unchanged','all_original_FC2_parameter_values_preserved',
            'FC2_weights_not_duplicated','FC2_bias_added_once_after_partial_sum','original_retained_and_target_output_bindings_protobuf_unchanged')
        require(all(proof[key] is True for key in required_flags) and proof['FC2_matmul_replacement_count']==10
            and proof['original_parameter_value_count']==180 and proof['FP32_reduction_order_changed'] is True
            and proof['partial_sum_order']==ORDER,'All original computation and K4 proof required')
        require(checked['schema']=='v73_original_VGGT_retained_FC2_K4_check'
            and checked['status']=='passed_original_FP32_ONNX_equivalence_not_quality_accepted'
            and checked['script_sha256']==BUILD_SHA and checked['candidate_onnx_sha256']==compiled['onnx']['sha256']
            and checked['baseline_manifest_sha256']==helper.MANIFEST_SHA and checked['original_parent_manifest_sha256']==rt.PARENT_SHA
            and checked['baseline_graph_sha256']==row['sha256'] and checked['output_bindings']==bindings
            and checked['cases']==compiled['cases'] and checked['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] is True
            and checked['cleanup_errors']==[] and checked['sessions_dropped'] is True,'Fixed passed candidate FP32 gate required')
        cases=checked['cases'];require([c['tag'] for c in cases]==['v53','v56-calib00'],'Two original real cases required')
        for new,old in zip(cases,original_check['cases']):
            require(all(new[key]==old[key] for key in ('tag','directory','input','capture_manifest_sha256'))
                and len(new['checks'])==2 and all(c['name']==binding['name'] and c['versus_baseline_FP32_ONNX']['strict_pass'] is True
                    and c['versus_baseline_FP32_ONNX']['outside_tolerance']==0 and c['versus_baseline_FP32_ONNX']['strict_rtol']==1e-4
                    and c['versus_baseline_FP32_ONNX']['strict_atol']==1e-5 for c,binding in zip(new['checks'],bindings)), 'Same-input fixed FP32 check required')
        artifact(candidate_dir,compiled['onnx']);candidate_model=artifact(candidate_dir,compiled['rknn'])
        require({4,5,6,7}<=old_affinity,'Original four CPUs unavailable');os.sched_setaffinity(0,{4,5,6,7})
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)),manifest_sha256=helper.MANIFEST_SHA,
            check_receipt_sha256=CHECK_SHA,check_result_sha256=CHECK_RESULT_SHA,build_source_sha256=BUILD_SHA,
            runtime_sha256=helper.RUNTIME_SHA,ABI_sha256=helper.ABI_SHA,library_sha256=helper.LIBRARY_SHA,NEON_sha256=helper.NEON_SHA)
        api=abi.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so',helper.LIBRARY_SHA))))
        neon=ct.CDLL(str(pin(BASE/'experiments/vggt-v56-neon-build-r1/fp16_neon.so',helper.NEON_SHA)))
        for name in ('f32_to_f16','f16_to_f32'):
            function=getattr(neon,name);function.argtypes,function.restype=[ct.c_void_p,ct.c_void_p,ct.c_size_t],None
        route=SimpleNamespace(base=abi,api=api,np=np,neon=neon,stats=stats,timeout=30000);routes={}
        for tag,model in [('A',original_model),('B',candidate_model)]:
            context=abi.CONTEXT();contexts.append(context);filename=ct.create_string_buffer(os.fsencode(model))
            code=api.rknn_init(ct.byref(context),ct.cast(filename,ct.c_void_p),0,0,None)
            report['contexts_created']+=int(bool(context.value));abi.check_call(code,'rknn_init');require(bool(context.value),'Empty SDK context')
            abi.check_call(api.rknn_set_core_mask(context,7),'rknn_set_core_mask')
            counts=abi.query(api,context,0,abi.InputOutputNum());require((counts.n_input,counts.n_output)==(1,2),'Two original outputs required')
            version=abi.query(api,context,5,abi.SDKVersion());require(re.match(r'^2\.3\.2(?:\D|$)',bytes(version.api_version).decode()),'SDK2.3.2 required')
            descriptors=[]
            for command,public in ((1,[dict(index=0,name='input',shape=list(rt.FRAME))]),(2,bindings)):
                for binding in public:
                    descriptors.append(rt.half_descriptor(abi,abi.query(api,context,command,abi.TensorAttr(index=binding['index'])),binding['shape'],binding['name'],binding['index']))
            memory=abi.query(api,context,6,abi.MemSize());routes[tag]=context
            report['queries'].append(dict(route=tag,model_sha256=sha(model),input=descriptors[0],outputs=descriptors[1:],
                SDK_version=bytes(version.api_version).decode(),driver_version=bytes(version.drv_version).decode(),
                weight_bytes=int(memory.total_weight_size),internal_bytes=int(memory.total_internal_size),dma_bytes=int(memory.total_dma_allocated_size)))
        for case in cases:
            x=np.load(artifact(candidate_dir,case['saved_input']),allow_pickle=False);artifact(case['directory'],case['input'])
            require(x.shape==rt.FRAME and x.dtype==np.float32 and x.flags.c_contiguous and np.isfinite(x).all()
                and sha_raw(x)==case['saved_input']['raw_sha256']==case['input']['raw_sha256'],'Pinned original FP32 input differs')
            references=[np.load(artifact(candidate_dir,item),allow_pickle=False) for item in (case['original_reference_outputs'] or [])]
            for value,binding in zip(references,bindings):require(value.shape==tuple(binding['shape']) and value.dtype==np.float32 and np.isfinite(value).all(),'Original Torch reference differs')
            first,seen={},dict(A=0,B=0)
            for index,tag in enumerate('ABBA'*6):
                before=helper.sample();require(before['npu_hz']==args.expected_npu_hz,'Unexpected frequency before SDK call')
                previous=dict(stats['half_io']);tick=time.perf_counter()
                values=rt.RKNNRetainedOriginalAggregator._run_group(route,routes[tag],x,bindings)
                elapsed=time.perf_counter()-tick
                after=helper.sample();require(after['npu_hz']==args.expected_npu_hz,'Unexpected frequency after SDK call')
                call=dict(index=len(report['calls']),case_index=index,input=case['tag'],route=tag,first_of_case_route=seen[tag]==0,
                    warm=seen[tag]>0,seconds=elapsed,timing={k:v-previous[k] for k,v in stats['half_io'].items()},
                    platform_before=before,platform_after=after,source_input=case['saved_input'],outputs=[])
                if tag not in first:
                    first[tag]=values;path=args.output_dir/f'first-{case["tag"]}-{tag}.npz'
                    with path.open('xb') as stream:np.savez(stream,**{f'output{i}':value for i,value in enumerate(values)})
                    call['artifact']=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))
                for i,(value,binding) in enumerate(zip(values,bindings)):
                    output=dict(index=i,name=binding['name'],shape=list(value.shape),dtype=str(value.dtype),tensor_bytes=value.nbytes,
                        raw_sha256=sha_raw(value),finite=True,owned=bool(value.flags.owndata),byte_equal_to_first_of_route=value.tobytes()==first[tag][i].tobytes())
                    if references:output['versus_original_Torch_FP32']=comparison(np,value,references[i])
                    call['outputs'].append(output)
                if 'A' in first:call['versus_first_A_same_input']=[comparison(np,value,reference) for value,reference in zip(values,first['A'])]
                seen[tag]+=1;report['calls'].append(call)
            require(sha_raw(x)==case['input']['raw_sha256'],'Original input changed during SDK calls')
        require(len(report['calls'])==48 and stats['sdk_run_attempts']==stats['sdk_runs_succeeded']==48
            and stats['outputs_acquired']==stats['outputs_released']==96,'48 SDK calls and96 releases required')
        summary={}
        for tag in ('A','B'):
            rows=[c for c in report['calls'] if c['route']==tag and c['warm']];times=np.array([c['seconds'] for c in rows])
            require(len(rows)==22,'22 warm per route required')
            summary[tag]=dict(warm_count=len(rows),mean_seconds=float(times.mean()),median_seconds=float(np.median(times)),
                p95_seconds=float(np.percentile(times,95)),min_seconds=float(times.min()),max_seconds=float(times.max()),
                inputs={case['tag']:dict(warm_count=11,mean_seconds=float(np.mean([c['seconds'] for c in rows if c['input']==case['tag']]))) for case in cases})
        report.update(status='completed_group0_component_comparison_not_quality_accepted',summary=summary)
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        for context in reversed(contexts):destroy(context)
        report['live_contexts']=sum(bool(c.value) for c in contexts);report['stats']=stats
        try:report['source_and_inputs_unchanged']=bool(pins) and all(sha(path)==digest for path,digest in pins.items())
        except BaseException as exc:report['source_and_inputs_unchanged']=False;report['cleanup_errors'].append('Identity: '+str(exc))
        try:os.sched_setaffinity(0,old_affinity);report['affinity_restored']=os.sched_getaffinity(0)==old_affinity
        except BaseException as exc:report['affinity_restored']=False;report['cleanup_errors'].append('Affinity: '+str(exc))
        if report['cleanup_errors'] or report['live_contexts'] or not report['source_and_inputs_unchanged'] or not report['affinity_restored']:
            report['status']='failed_cleanup_or_identity'
        temp=args.output_dir/'result.json.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(args.output_dir/'result.json')
        print(json.dumps({key:report.get(key) for key in ('status','error','contexts_created','contexts_destroyed','cleanup_errors')}),flush=True)
    return 0 if report['status']=='completed_group0_component_comparison_not_quality_accepted' else 1


def sha_raw(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


if __name__=='__main__':sys.exit(main())
