#!/usr/bin/env python3
"""Board-only shared CameraHead RKNN context, four sequential refinement steps.

Actual SDK tensor names bind both inputs, independent of SDK input ordering.
Every delta is copied before SDK release; CPU FP32 additions and final original
linear/linear/ReLU activation remain explicit. One first head plus three timed
subsequent heads use the same context/weights. Profiling is a separate run.
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
import re
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
EXPORT_SHA = 'b26efe9c6cd62a2b7977e6e9c096cf0bd9bbf63076a063b9a332ef59565af258'
CPUS = {6, 7}


def load_frozen(name, digest, module_name):
    path = Path(__file__).with_name(name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Frozen helper SHA mismatch: {name}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def validate_bindings(base, inputs, output):
    expected = {'camera_tokens': (1, 2, 2048), 'conditioning_pose': (1, 2, 9)}
    bindings = {}
    for attr in inputs:
        name = bytes(attr.name).decode('utf-8')
        if name not in expected or name in bindings:
            raise ValueError('Exactly one raw-token and one conditioning SDK input required')
        previous = base.SHAPE
        try:
            base.SHAPE = expected[name]
            base.validate_attr(attr)
        finally:
            base.SHAPE = previous
        bindings[name] = {'index': int(attr.index), 'format': int(attr.fmt)}
    if set(bindings) != set(expected) or {v['index'] for v in bindings.values()} != {0, 1}:
        raise ValueError('SDK input names or indices are incomplete/duplicated')
    if bytes(output.name).decode('utf-8') != 'pose_delta' or output.index != 0:
        raise ValueError('Single SDK output must be pose_delta at index zero')
    previous = base.SHAPE
    try:
        base.SHAPE = (1, 2, 9)
        base.validate_attr(output)
    finally:
        base.SHAPE = previous
    return bindings


def execute_step(base, api, context, raw, conditioning, bindings, timeout, cleanup, activity):
    import numpy as np
    request, returned = (base.Input * 2)(), (base.Output * 1)()
    for name,array in [('camera_tokens',raw),('conditioning_pose',conditioning)]:
        binding = bindings[name]
        request[binding['index']] = base.Input(binding['index'], array.ctypes.data, array.nbytes,
                                              0, 0, binding['format'])
    returned[0].want_float = 1
    run, got, receive = base.RunExtend(0,0,timeout,-1), False, base.OutputExtend()
    row, begin = dict(synchronous=True, input_type='FP32', input_pass_through=0, want_float=1), time.perf_counter()
    try:
        base.check_call(api.rknn_inputs_set(context,2,request), 'rknn_inputs_set')
        set_end = time.perf_counter()
        activity['npu_run_api_called'] = True
        activity['npu_run_api_call_count'] += 1
        base.check_call(api.rknn_run(context,ct.byref(run)), 'rknn_run')
        run_end = time.perf_counter()
        base.check_call(api.rknn_outputs_get(context,1,returned,ct.byref(receive)), 'rknn_outputs_get')
        got, get_end = True, time.perf_counter()
        if run.frame_id != receive.frame_id:
            raise ValueError('Synchronous RKNN step frame identity mismatch')
        if not returned[0].buf or returned[0].size != 18 * 4:
            raise ValueError('SDK delta must be 18 owned-convertible FP32 values')
        array = np.ctypeslib.as_array(ct.cast(returned[0].buf,ct.POINTER(ct.c_float)),shape=(18,))
        owned = array.copy().reshape(1,2,9)
        copy_end = time.perf_counter()
        row.update(input_set_seconds=set_end-begin, run_api_seconds=run_end-set_end,
                   outputs_get_seconds=get_end-run_end, output_copy_seconds=copy_end-get_end,
                   runtime_frame_id=int(run.frame_id), output_bytes=int(returned[0].size))
    finally:
        if got or returned[0].buf:
            tick = time.perf_counter()
            code = api.rknn_outputs_release(context,1,returned)
            end = time.perf_counter()
            row.update(outputs_release_seconds=end-tick, output_buffer_released=code==0, io_total_seconds=end-begin)
            if code:
                cleanup.append(f'rknn_outputs_release returned {code}')
            base.check_call(code, 'rknn_outputs_release')
    if owned.dtype != np.float32 or not np.isfinite(owned).all():
        raise ValueError('Nonfinite RKNN delta rejected')
    return owned,row


def load_artifacts(args, base, step):
    import numpy as np
    old,old_path,capture,capture_path = step.load_frozen()
    helper,helper_path = capture.base_capture()
    cap_path = args.capture_dir / 'capture.json'
    pins = {str(cap_path):step.require_sha(cap_path,args.capture_sha256),
            str(args.export_receipt):step.require_sha(args.export_receipt,args.export_sha256),
            str(args.compile_receipt):step.require_sha(args.compile_receipt,args.compile_sha256),
            str(Path(__file__).resolve()):step.sha(__file__),
            str(old_path):step.EXPORT_V15_SHA,str(capture_path):step.CAPTURE_SHA,
            str(helper_path):capture.BASE_CAPTURE_SHA}
    cap,exported,compiled = [json.loads(path.read_text()) for path in (cap_path,args.export_receipt,args.compile_receipt)]
    if (cap.get('status') != 'completed' or cap.get('script_sha256') != step.CAPTURE_SHA
            or cap.get('npu_used') is not False or cap.get('cleanup_errors')
            or cap.get('source_and_inputs_unchanged') is not True
            or cap.get('checkpoint',{}).get('sha256') != capture.CHECKPOINT_SHA
            or exported.get('script_sha256') != EXPORT_SHA or exported.get('cleanup_errors')
            or exported.get('status') != 'exported_and_structurally_checked'
            or exported.get('capture',{}).get('sha256') != args.capture_sha256
            or exported.get('source_and_inputs_unchanged') is not True
            or exported.get('weight_sharing',{}).get('original_parameters_verified') != 68
            or exported.get('outer_iteration_contract',{}).get('all_four_original_iterations_array_equal') is not True
            or compiled.get('status') != 'compiled_not_runtime_validated'
            or compiled.get('export_receipt_sha256') != args.export_sha256
            or compiled.get('exporter_script_sha256') != EXPORT_SHA
            or compiled.get('input',{}).get('sha256') != exported.get('onnx',{}).get('sha256')
            or compiled.get('output',{}).get('sha256') != args.model_sha256
            or compiled.get('wrapper_and_frozen_compiler_unchanged') is not True):
        raise ValueError('Original capture -> shared-step ONNX -> RKNN SHA/provenance mismatch')
    originals = {**helper.SOURCE_PINS,**capture.EXTRA_PINS}
    if cap.get('source_hashes') != originals:
        raise ValueError('Original model source hashes differ')
    for name,digest in originals.items():
        pins[str(SOURCE/name)] = step.require_sha(SOURCE/name,digest)
    arrays = {}
    for name,shape in [('input.npy',step.INPUT_SHAPE),('output.npy',step.POSE_SHAPE)]:
        path,item = args.capture_dir/name,cap['artifacts'][name]
        pins[str(path)] = step.require_sha(path,item['sha256'])
        array = np.load(path,allow_pickle=False)
        if (array.shape != shape or array.dtype != np.float32 or not array.flags.c_contiguous
                or not np.isfinite(array).all() or path.stat().st_size != item['bytes']
                or helper.array_info(array)['sha256'] != item['tensor_sha256']):
            raise ValueError('Exact original FP32 tokens/reference required')
        arrays[name] = array
    export_dir = args.export_receipt.parent
    for name in ('conditioning-seed.npy','empty-pose-token.npy','step-iterations.npz'):
        item,path = exported['artifacts'][name],export_dir/name
        pins[str(path)] = step.require_sha(path,item['sha256'])
        if path.stat().st_size != item['bytes']:
            raise ValueError('Step reference artifact byte mismatch')
    seed = np.load(export_dir/'conditioning-seed.npy',allow_pickle=False)
    empty = np.load(export_dir/'empty-pose-token.npy',allow_pickle=False)
    if (helper.array_info(empty) != cap['artifacts']['camera-state.npz']['tensors']['empty_pose_tokens']
            or not np.array_equal(seed,np.broadcast_to(empty,step.POSE_SHAPE))):
        raise ValueError('First conditioning must be the original learned empty pose, expanded')
    step.pose_array(seed)
    with np.load(export_dir/'step-iterations.npz',allow_pickle=False) as saved:
        references = {name:saved[name].copy() for name in saved.files}
    expected_names = {f'{kind}_{i}' for kind in ('conditioning','delta','preactivation','activation') for i in range(4)}
    if set(references) != expected_names:
        raise ValueError('All four original iteration references required')
    for key,array in references.items():
        step.pose_array(array)
        if helper.array_info(array) != exported['artifacts']['step-iterations.npz']['tensors'][key]:
            raise ValueError('Exact step reference tensor SHA mismatch')
    if not np.array_equal(references['activation_3'],arrays['output.npy']):
        raise ValueError('Step final reference differs from original full head')
    return arrays['input.npy'],arrays['output.npy'],seed,references,cap,exported,compiled,pins


def targets(detail):
    result = []
    for line in detail['raw_text'].splitlines():
        match = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU)\s+',line)
        if match:
            result.append(dict(id=int(match[1]),operator_type=match[2],dtype=match[3],target=match[4]))
    return result


def run(args,base,step):
    report = dict(kind='original_shared_CameraHead_four_step_RKNN_board_probe',status='failed',
                  script_sha256=step.sha(__file__),frozen_runtime_sha256=RUNTIME_SHA,
                  camera_iterations=4,trunk_blocks_per_step=4,one_context_for_all_steps=True,
                  npu_run_api_called=False,npu_run_api_call_count=0,ORB_used=False,production_modified=False,
                  full_model_quality_validated=False,real_time_validated=False,all_operators_on_npu_verified=False,
                  core_mask=args.core_mask,threads=2,cpus_requested=sorted(CPUS),init_flags=8 if args.collect_perf_detail else 0,
                  perf_detail_collection_enabled=args.collect_perf_detail,normal_benchmark_timing=not args.collect_perf_detail,
                  first_head_count=1,subsequent_measured_head_count=3,unmeasured_warmup_calls=0,
                  calls=[],artifacts=[],cleanup_errors=[],abi=base.abi_record())
    allowed,context,api,pins,start = None,base.CONTEXT(),None,{},time.monotonic()
    try:
        report['device_tree_compatible'] = base.require_board()
        allowed = set(os.sched_getaffinity(0))
        if not CPUS <= allowed:
            raise RuntimeError('CPUs 6,7 must be allowed')
        os.sched_setaffinity(0,CPUS)
        os.environ.update({key:'2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        import numpy as np
        raw,expected,seed,references,cap,exported,compiled,pins = load_artifacts(args,base,step)
        if not 0 < args.model.stat().st_size <= 1024**3:
            raise ValueError('Shared-step compiled model must fit the 1GiB model-file budget')
        pins[str(args.model)] = step.require_sha(args.model,args.model_sha256)
        header = Path(__file__).resolve().parents[1]/'reference/rknn_api_v2.3.2.h'
        pins[str(header)] = step.require_sha(header,base.HEADER_SHA256)
        for name,digest in [('export-v16-camera-step-board.py',EXPORT_SHA),('run-v12-rknn-block-board.py',RUNTIME_SHA)]:
            path = Path(__file__).with_name(name)
            pins[str(path)] = step.require_sha(path,digest)
        library = args.library.resolve(strict=True)
        pins[str(library)] = step.sha(library)
        report.update(source_and_input_hashes=pins.copy(),capture_sha256=args.capture_sha256,
                      export_sha256=args.export_sha256,compile_sha256=args.compile_sha256,
                      checkpoint=cap['checkpoint'],compiled_model_bytes=args.model.stat().st_size,
                      original_step_parameter_bytes=exported['weight_sharing']['original_unique_parameter_bytes'],
                      input_shapes=[list(raw.shape),list(seed.shape)],output_shape=[1,2,9],
                      cpu_affinity=sorted(os.sched_getaffinity(0)),memory_before_sdk=base.process_memory())
        api = base.bind_api(ct.CDLL(str(library)))
        tick = time.perf_counter()
        filename = ct.create_string_buffer(os.fsencode(args.model.resolve()))
        base.check_call(api.rknn_init(ct.byref(context),ct.cast(filename,ct.c_void_p),0,report['init_flags'],None),'rknn_init')
        report['init_seconds'] = time.perf_counter()-tick
        base.check_call(api.rknn_set_core_mask(context,args.core_mask),'rknn_set_core_mask')
        report['core_mask_set_successfully'] = True
        version = base.query(api,context,5,base.SDKVersion())
        report.update(sdk_version=bytes(version.api_version).decode(errors='replace'),driver_version=bytes(version.drv_version).decode(errors='replace'))
        if not report['sdk_version'].startswith('2.3.2'):
            raise ValueError('Exact RKNN runtime 2.3.2 ABI required')
        counts = base.query(api,context,0,base.InputOutputNum())
        report['input_output_num'] = dict(n_input=int(counts.n_input),n_output=int(counts.n_output))
        if counts.n_input != 2 or counts.n_output != 1:
            raise ValueError('Exactly two SDK inputs and one delta output required')
        attrs = [base.query(api,context,1,base.TensorAttr(index=i)) for i in range(2)]
        attr = base.query(api,context,2,base.TensorAttr(index=0))
        report.update(input_attrs=[base.attr_record(a) for a in attrs],output_attr=base.attr_record(attr),tensor_attributes_validated=False)
        bindings = validate_bindings(base,attrs,attr)
        report.update(actual_SDK_input_bindings=bindings,tensor_attributes_validated=True)
        memory = base.MemSize()
        code = api.rknn_query(context,6,ct.byref(memory),ct.sizeof(memory))
        report.update(model_memory_query_code=int(code),model_memory_bytes={name:int(getattr(memory,name)) for name,_ in memory._fields_ if name!='reserved'} if code==0 else None)
        first = None
        for index in range(4):
            prediction,steps,arrays = None,[],{}
            head_begin,scalar_seconds = time.perf_counter(),0.
            for i in range(4):
                conditioning = seed if prediction is None else prediction
                delta,row = execute_step(base,api,context,raw,conditioning,bindings,args.run_timeout_ms,report['cleanup_errors'],report)
                tick = time.perf_counter()
                prediction = step.accumulate_delta(prediction,delta)
                scalar_seconds += time.perf_counter()-tick
                row['iteration'] = i
                # Profiling is a separate diagnostic run, not normal timing.
                if args.collect_perf_detail:
                    row['perf_detail'] = base.perf_detail(api,context)
                    row['operator_targets'] = targets(row['perf_detail'])
                steps.append(row)
                arrays.update({f'conditioning_{i}':conditioning.copy(),f'delta_{i}':delta.copy(),f'preactivation_{i}':prediction.copy()})
            tick = time.perf_counter()
            actual = step.activate_numpy_pose(prediction)
            scalar_seconds += time.perf_counter()-tick
            whole_seconds = time.perf_counter()-head_begin
            reference_steps = []
            for i in range(4):
                arrays[f'activation_{i}'] = step.activate_numpy_pose(arrays[f'preactivation_{i}'])
                reference_steps.append(dict(iteration=i,conditioning_vs_original_cpu=base.comparison(arrays[f'conditioning_{i}'],references[f'conditioning_{i}']),
                    delta_vs_original_cpu=base.comparison(arrays[f'delta_{i}'],references[f'delta_{i}']),
                    activation_vs_original_cpu=base.comparison(arrays[f'activation_{i}'],references[f'activation_{i}']),
                    delta_input_equals_CPU_reference=bool(np.array_equal(arrays[f'conditioning_{i}'],references[f'conditioning_{i}']))))
            if first is None:
                first = actual.copy()
            row = dict(call_index=index,phase='first' if index==0 else 'subsequent',whole_head_seconds=whole_seconds,
                       summed_step_IO_seconds=sum(x['io_total_seconds'] for x in steps),CPU_FP32_accumulation_and_final_activation_seconds=scalar_seconds,
                       steps=steps,versus_cpu_reference=base.comparison(actual,expected),versus_first_head=base.comparison(actual,first),
                       iterations_vs_original_cpu=reference_steps)
            for name,data in [(f'output-{index:02d}.npy',actual),(f'head-iterations-{index:02d}.npz',arrays)]:
                path = args.output_dir/name
                with path.open('xb') as stream:
                    if isinstance(data,dict):
                        np.savez(stream,**data)
                    else:
                        np.save(stream,data,allow_pickle=False)
                report['artifacts'].append(dict(file=name,sha256=step.sha(path),bytes=path.stat().st_size))
            report['calls'].append(row)
        warm = [row['whole_head_seconds'] for row in report['calls'][1:]]
        report['subsequent_whole_head_seconds'] = dict(values=warm,mean=float(np.mean(warm)),min=min(warm),max=max(warm))
        report['outer_iteration_contract'] = exported['outer_iteration_contract']
        report['timing_limitations'] = ['Four heads are not statistical or whole-model/capture-to-pose validation',
            'Whole-head timing includes four synchronous SDK IO steps, small-array copies, CPU FP32 recurrence and final activation; excludes artifact writes/comparisons/hashing',
            'Default run makes no CPU model calls; original CPU reference was captured separately',
            'Profile mode includes SDK detail queries and is not a normal performance benchmark',
            'Only actual SDK-reported targets establish graph operator placement; CPU IO conversion/recurrence/activation are explicit']
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if api is not None and context.value:
            try:
                base.check_call(api.rknn_destroy(context),'rknn_destroy')
                report['context_destroyed'] = True
            except BaseException as exc:
                report['cleanup_errors'].append(f'Context cleanup: {exc}')
        else:
            report['context_destroyed'] = None
        if pins:
            try:
                report['source_and_input_hashes_after'] = {path:step.sha(path) for path in pins}
                report['source_and_inputs_unchanged'] = report['source_and_input_hashes_after']==pins
                if not report['source_and_inputs_unchanged']:
                    report['cleanup_errors'].append('Input/source/runtime changed')
            except BaseException as exc:
                report['cleanup_errors'].append(f'Source recheck: {exc}')
        if allowed is not None:
            try:
                os.sched_setaffinity(0,allowed)
                report['affinity_restored'] = set(os.sched_getaffinity(0))==allowed
                if not report['affinity_restored']:
                    raise RuntimeError('Affinity restoration mismatch')
            except BaseException as exc:
                report['cleanup_errors'].append(f'Affinity cleanup: {exc}')
        report.update(successful_head_call_count=len(report['calls']),memory_final=base.process_memory(),
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,total_seconds=time.monotonic()-start)
        if report['cleanup_errors']:
            report['status'] = 'failed'
        with (args.output_dir/'receipt.json').open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
            stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','export-receipt','compile-receipt','capture-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('model-sha256','export-sha256','compile-sha256','capture-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'))
    parser.add_argument('--core-mask',type=int,choices=(1,7),default=7)
    parser.add_argument('--run-timeout-ms',type=int,default=30000)
    parser.add_argument('--collect-perf-detail',action='store_true')
    args = parser.parse_args()
    base,_ = load_frozen('run-v12-rknn-block-board.py',RUNTIME_SHA,'_v16_RKNN_ABI')
    step,_ = load_frozen('export-v16-camera-step-board.py',EXPORT_SHA,'_v16_step_contract')
    base.require_board()
    if not 1<=args.run_timeout_ms<=60000:
        parser.error('run-timeout-ms must be within 1..60000')
    if Path.cwd().resolve()!=BASE or Path(__file__).resolve().parents[3]!=SOURCE:
        parser.error('Run deployed source from board BASE')
    if not args.output_dir.resolve().is_relative_to(step.EXPERIMENT) or os.path.lexists(args.output_dir):
        parser.error('Use a NEW output directory below experiments/vggt-v16-camera-step')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    result = run(args,base,step)
    path = args.output_dir/'receipt.json'
    print(json.dumps(dict(status=result['status'],error=result.get('error'),output=str(path),sha256=step.sha(path))))
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    sys.exit(main())
