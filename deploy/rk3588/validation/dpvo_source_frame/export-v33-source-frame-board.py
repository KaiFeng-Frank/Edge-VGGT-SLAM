#!/usr/bin/env python3
"""Board-only complete-source DPVO Update export and original/ORT comparison.

Uses two frozen real E23520 recurrent states, selects the first minimum-degree
complete source, and keeps strict numerical rejection independent of successful
diagnostic execution. No encoder, geometry, new frame or pose is executed.
"""
import argparse,hashlib,importlib.util,inspect,json,os,platform,resource,sys,time,traceback
from pathlib import Path

BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
LARGE_SHA='25ed1a743bbda78b8fa8eeefb0ed4827ca1c3d2fdf978c0c1e2a294e97ef3520'
PLAN_SHA='04e240c2c35bd1359dbe5649fb038490a0a7ff99654d8d112ad7c83acc87ac2e'
CHECKPOINT_SHA='30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
PINS={'dpvo_probe_common.py':'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9',
      'dpvo_cpu_ops.py':'15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32',
      'manifest.json':'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375'}
INPUTS=('net','inp','corr');OUTPUTS=('net_out','delta','weight')

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):value.update(chunk)
    return value.hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def meta(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
def compare(actual,reference):
    import numpy as np
    require(actual.dtype==reference.dtype==np.float32 and actual.shape==reference.shape,'Matching FP32 shapes required')
    require(np.isfinite(actual).all() and np.isfinite(reference).all(),'Nonfinite actual/reference')
    difference=actual.astype(np.float64)-reference.astype(np.float64)
    bad=int(np.count_nonzero(np.abs(difference)>1e-5+1e-4*np.abs(reference.astype(np.float64))))
    norm=float(np.linalg.norm(reference.astype(np.float64).ravel()))
    return dict(rtol=1e-4,atol=1e-5,failed_elements=bad,gate_passed=bad==0,
                max_abs=float(np.abs(difference).max()),relative_l2=float(np.linalg.norm(difference.ravel()))/norm if norm else None,
                bitwise_equal=actual.tobytes(order='C')==reference.tobytes(order='C'))
def require_outputs(values,degree,patches):
    import numpy as np
    require(tuple(values)==OUTPUTS,'Ordered three outputs required')
    for name,channels in zip(OUTPUTS,(384,2,2)):
        value=values[name];require(value.dtype==np.float32 and value.shape==(1,degree,patches,channels),'Output contract differs: '+name)
        require(np.isfinite(value).all(),'Nonfinite output: '+name)
    weight=values['weight'];require(((weight>=0)&(weight<=1)).all(),'Confidence outside [0,1]')
    return dict(outputs_finite=True,weight_domain_valid=True,weight_min=float(weight.min()),weight_max=float(weight.max()))
def model_contract(value):
    tensor=value.type.tensor_type
    require(tensor.elem_type==1 and all(d.HasField('dim_value') and d.dim_value>0 for d in tensor.shape.dim),'Fixed FP32 ONNX contract required')
    return dict(name=value.name,dtype='float32',shape=[d.dim_value for d in tensor.shape.dim])
def initializer_coverage(model,parameters,onnx):
    """Bind every learned tensor to original bytes or its exact 2D transpose."""
    originals={}
    for name,tensor in parameters.items():
        value=tensor.detach().numpy();choices=[('identity',value)]
        if value.ndim==2:choices.append(('transpose',value.T))
        for transform,array in choices:
            key=(tuple(array.shape),str(array.dtype),hashlib.sha256(array.tobytes(order='C')).hexdigest())
            originals.setdefault(key,[]).append(dict(parameter=name,transform=transform))
    rows=[];covered=set()
    for tensor in model.graph.initializer:
        require(tensor.data_location!=onnx.TensorProto.EXTERNAL and not tensor.external_data,'External initializer data prohibited')
        value=onnx.numpy_helper.to_array(tensor);key=(tuple(value.shape),str(value.dtype),hashlib.sha256(value.tobytes(order='C')).hexdigest())
        matches=originals.get(key,[]);require(bool(matches),'Unbound ONNX initializer '+tensor.name)
        covered.update(m['parameter'] for m in matches);rows.append(dict(initializer=tensor.name,array=meta(value),original_matches=matches))
    require(covered==set(parameters),'Original Update parameter coverage differs')
    return dict(all_original_parameters_covered=True,original_tensor_count=len(parameters),initializer_count=len(rows),mapping=rows)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('v26-source-dir','checkpoint','large-dir','wrapper-source','plan-source','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wrapper-sha256',required=True);args=parser.parse_args(argv)
    require(platform.system()=='Linux' and platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'Actual model export/inference is RK3588-only')
    out=args.output_dir.resolve();require(out.is_relative_to(BASE) and out!=BASE,'Isolated experiment path required')
    protected=[v.resolve() for k,v in vars(args).items() if k!='output_dir' and isinstance(v,Path)]
    require(not any(out==p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected),'Output overlaps source/input')
    out.mkdir(exist_ok=False);started=time.monotonic();affinity=os.sched_getaffinity(0);pins={};neural=None;before=None;session=None
    report=dict(schema='v33_DPVO_source_frame_Update_board_export',status='failed',script_sha256=sha(__file__),
                actual_board_execution=True,actual_inference_on_RK3588S=True,host_model_inference=False,
                original_VGGT_default_unchanged=True,different_model_from_VGGT=True,ORB_used=False,added_LK_PnP_frontend=False,
                NPU_used=False,BA_executed=False,valid_pose_output=False,realtime_validated=False,trajectory_accuracy_validated=False,
                candidate_accepted=False,strict_FP32_gate_passed=False,outputs_finite=False,weight_domain_valid=False,
                original_parameters_unchanged=False,cases=[],cleanup_errors=[],terminal_sigmoid='stable',
                scope='Two fixed real recurrent inputs; one complete source block per state; no new observation, BA or candidate feedback.')
    def pin(path,expected):
        path=Path(path).resolve();require(sha(path)==expected,'Pinned source/input differs: '+str(path));pins[str(path)]=expected;return path
    def load_saved(folder,item):
        require(Path(item['file']).name==item['file'],'Artifact basename required')
        path=pin(folder/item['file'],item['sha256']);require(path.stat().st_size==item['bytes'],'Artifact size differs')
        with np.load(path,allow_pickle=False) as saved:values={k:saved[k].copy() for k in saved.files}
        require({k:meta(v) for k,v in values.items()}==item['arrays'],'Saved raw-array identity differs');return values
    def save(name,values):
        path=out/name
        with path.open('xb') as stream:np.savez(stream,**values)
        return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in values.items()})
    try:
        pin(__file__,report['script_sha256']);pin(args.wrapper_source,args.wrapper_sha256);pin(args.plan_source,PLAN_SHA);pin(args.checkpoint,CHECKPOINT_SHA)
        for name,digest in PINS.items():pin(args.v26_source_dir/name,digest)
        old=json.loads(pin(args.large_dir/'run.json',LARGE_SHA).read_text())
        require(old['status']=='completed_numerical_rejected' and old['original_weights_unchanged'] and old['sources_and_inputs_unchanged'] and not old['cleanup_errors'],'Preserved original large-state provenance required')
        require(len(old['states'])==2,'Both predeclared real states required')
        require({4,5,6,7}<=affinity,'Four A76 CPUs required');os.sched_setaffinity(0,{4,5,6,7})
        import numpy as np
        import torch
        import onnx
        import onnxruntime as ort
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        common=module(args.v26_source_dir/'dpvo_probe_common.py','_v33_original_loader')
        layout=module(args.plan_source,'_v33_frozen_source_plan');wrapper=module(args.wrapper_source,'_v33_source_update')
        neural,loader=common.load_neural(args.v26_source_dir/'official',args.v26_source_dir/'manifest.json',args.checkpoint,CHECKPOINT_SHA,
                                        args.v26_source_dir/'dpvo_cpu_ops.py',cpu_ops_sha256=PINS['dpvo_cpu_ops.py'])
        for name,item in loader['source_hashes'].items():pin(args.v26_source_dir/'official'/name,item['sha256'])
        before={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
        require(before==old['original_weights_before'] and len(before)==94,'Original parameters differ from saved real-state run')
        wrapped=wrapper.SourceFrameUpdate(neural.update,terminal_sigmoid='stable').eval()
        report.update(loader=loader,original_weights_before=before,original_Update_parameters={k:meta(v.detach().numpy()) for k,v in neural.update.state_dict().items()},
                      original_binding=wrapped.verify_original_binding(neural.update),cpu_affinity=sorted(os.sched_getaffinity(0)),
                      torch_threads=4,torch_interop_threads=1,torch_version=torch.__version__,numpy_version=np.__version__,onnx_version=onnx.__version__,ORT_version=ort.__version__)
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry('session.intra_op.allow_spinning','0');options.add_session_config_entry('session.inter_op.allow_spinning','0')
        all_gates=[];input_contract=None;output_contract=None
        for case_index,state in enumerate(old['states']):
            require(state['edges']==23520,'Expected real large graph')
            values=load_saved(args.large_dir,state['inputs_artifact']);historical=load_saved(args.large_dir,state['representative_outputs']['Torch'])
            plans=layout.build_source_frame_plan(*(values[k] for k in ('ii','jj','kk')),patches_per_frame=96,frame_capacity=64,max_target_degree=25)
            plan=min(plans,key=lambda p:(p.tensor_shape_without_batch_channels[0],p.source_slot))
            degree,patches=plan.tensor_shape_without_batch_channels;require((degree,patches)==(10,96),'Predeclared minimum source shape changed')
            feed={k:layout.to_canonical(values[oldkey],plan) for k,oldkey in zip(INPUTS,('net','context','corr'))}
            arguments=tuple(torch.from_numpy(feed[k]) for k in INPUTS);wrapped.validate_inputs(*arguments)
            case=dict(index=case_index,role=state['role'],source_plan=plan.to_metadata(),total_graph_edges=23520,all_source_blocks=len(plans),
                      original_input_artifact_sha256=state['inputs_artifact']['sha256'],inputs_artifact=save('case-%d-inputs.npz'%case_index,feed),
                      graph_artifact=save('case-%d-plan.npz'%case_index,dict(ii=values['ii'],jj=values['jj'],kk=values['kk'],canonical_edges=plan.canonical_edge_indices)))
            report['cases'].append(case)
            with torch.inference_mode():
                t=time.perf_counter();full_net,rest=neural.update(*(torch.from_numpy(values[k]) for k in ('net','context','corr')),None,*(torch.from_numpy(values[k]) for k in ('ii','jj','kk')))
                full={k:v.detach().numpy().copy() for k,v in zip(OUTPUTS,(full_net,rest[0],rest[1]))};case['original_full_Update_seconds']=time.perf_counter()-t
                case['fresh_full_reference_vs_historical']={k:compare(full[k],historical[oldkey]) for k,oldkey in zip(OUTPUTS,('net','delta','weight'))}
                require(all(x['gate_passed'] for x in case['fresh_full_reference_vs_historical'].values()),'Fresh original full reference differs from historical input/state')
                reference={k:layout.to_canonical(v,plan) for k,v in full.items()};require_outputs(reference,degree,patches)
                case['reference_artifact']=save('case-%d-reference.npz'%case_index,reference)
                t=time.perf_counter();result=wrapped(*arguments);candidate={k:v.detach().numpy().copy() for k,v in zip(OUTPUTS,result)};case['Torch_block_seconds']=time.perf_counter()-t
                case['wrapper_artifact']=save('case-%d-wrapper.npz'%case_index,candidate);case['wrapper_domain']=require_outputs(candidate,degree,patches)
                case['wrapper_vs_original_full']={k:compare(candidate[k],reference[k]) for k in OUTPUTS}
                all_gates.extend(x['gate_passed'] for x in case['wrapper_vs_original_full'].values())
                if case_index==0:
                    path=out/'update-d10-m96.onnx';export_options=dict(input_names=list(INPUTS),output_names=list(OUTPUTS),opset_version=17,do_constant_folding=True)
                    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:export_options['dynamo']=False
                    t=time.perf_counter();torch.onnx.export(wrapped,arguments,str(path),**export_options);report['export_seconds']=time.perf_counter()-t
                    model=onnx.load(path,load_external_data=False);onnx.checker.check_model(model,full_check=True)
                    report['parameter_coverage']=initializer_coverage(model,neural.update.state_dict(),onnx)
                    input_contract=[model_contract(v) for v in model.graph.input];output_contract=[model_contract(v) for v in model.graph.output]
                    require(input_contract==[dict(name=k,dtype='float32',shape=list(feed[k].shape)) for k in INPUTS],'Three exact model inputs required')
                    require(output_contract==[dict(name=k,dtype='float32',shape=list(reference[k].shape)) for k in OUTPUTS],'Three exact model outputs required')
                    operations=sorted({n.op_type for n in model.graph.node});require(not {'Gather','GatherElements','GatherND','Where'}&set(operations),'Unexpected generic group selection in graph')
                    report.update(inputs=input_contract,outputs=output_contract,model=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)),
                                  onnx_operations=operations,onnx_nodes=len(model.graph.node),group_Gather_Where_absent=True)
                    t=time.perf_counter();session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']);report['ORT_session_seconds']=time.perf_counter()-t
                    require(session.get_providers()==['CPUExecutionProvider'],'Only CPU ORT required');report['ORT_providers']=session.get_providers()
                    report['ORT_graph_optimization']='ORT_DISABLE_ALL'
            t=time.perf_counter();ort_values={k:np.array(v,copy=True,order='C') for k,v in zip(OUTPUTS,session.run(list(OUTPUTS),feed))};case['ORT_block_seconds']=time.perf_counter()-t
            case['ort_artifact']=save('case-%d-ort.npz'%case_index,ort_values);case['ort_domain']=require_outputs(ort_values,degree,patches)
            case['ORT_vs_original_full']={k:compare(ort_values[k],reference[k]) for k in OUTPUTS}
            case['ORT_vs_wrapper']={k:compare(ort_values[k],candidate[k]) for k in OUTPUTS}
            all_gates.extend(x['gate_passed'] for group in ('ORT_vs_original_full','ORT_vs_wrapper') for x in case[group].values())
            require({k:meta(feed[k]) for k in INPUTS}==case['inputs_artifact']['arrays'],'Block input changed')
            wrapped.verify_original_binding(neural.update)
            del values,historical,full,full_net,rest,result,candidate,ort_values,arguments
        report.update(status='completed_source_frame_export_diagnostic_not_adopted',strict_FP32_gate_passed=all(all_gates),
                      outputs_finite=True,weight_domain_valid=True,FP32_thresholds_relaxed=False,output_clipping=False,
                      timing_scope='First component calls, not a warm benchmark. Full original graph and one source block have different work; no speedup inferred.',
                      limitations=['Two frozen recurrent inputs, no independent candidate recurrence or trajectory.',
                                   'Single-source closed dependency block, not the whole online graph or a new pose.',
                                   'Source permutation and reduction/GEMM shape changes may alter rounding. Strict failures preserved.',
                                   'No host model execution, NPU execution, physical camera, dense map or real-time acceptance.'])
    except Exception as error:report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        session=None;report['ORT_session_reference_released']=True
        if neural is not None and before is not None:
            try:
                after={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()};report['original_weights_after']=after
                report['original_parameters_unchanged']=after==before;require(after==before,'Original weights changed')
            except Exception as error:report['cleanup_errors'].append(repr(error));report['status']='failed'
        try:
            report['source_hashes_after']={p:sha(p) for p in pins};report['sources_and_inputs_unchanged']=report['source_hashes_after']==pins
            require(report['sources_and_inputs_unchanged'],'Source/input changed')
        except Exception as error:report['cleanup_errors'].append(repr(error));report['status']='failed'
        try:os.sched_setaffinity(0,affinity);report['affinity_restored']=set(os.sched_getaffinity(0))==set(affinity);require(report['affinity_restored'],'Affinity restore failed')
        except Exception as error:report['cleanup_errors'].append(repr(error));report['status']='failed'
        report.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,actual_driver_seconds=time.monotonic()-started)
        (out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','strict_FP32_gate_passed','outputs_finite','weight_domain_valid','cleanup_errors')}),flush=True)
    return 0 if report['status']=='completed_source_frame_export_diagnostic_not_adopted' else 1

if __name__=='__main__':raise SystemExit(main())
