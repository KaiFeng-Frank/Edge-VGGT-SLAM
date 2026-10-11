#!/usr/bin/env python3
"""Board-only original FP32 DPVO Update export with dynamic E/group axes.

This exports standard ONNX, not a dynamically supported RKNN artifact. Only
the pinned E96 trace example is evaluated here; a separate board ORT run must
validate new E/G/L values. The frozen v29 terminal sigmoid identity changes no
weights or other Update nodes. No encoder, image, BA or pose is executed.
DPVO is MIT licensed, Copyright (c) 2022 Princeton Vision & Learning Lab.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback


BASE = Path('/mnt/vggt-data/vggt-slam2/experiments')
CAPTURE_SHA = '87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
CHECKPOINT_SHA = '30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
ADAPTER_SHA = '3f755c8e24d092fa55fdf698d87c3d9ba8944487ee95d4e5afc47554d31d1380'
REWRITER_SHA = '34bf40d845acecbadd43c95e01363370d133aab122451a105729802d5c378f37'
HISTORICAL_SIGMOID_ORT_SHA = '98b5e47b8558cc4440945e882729940464366e0fd27c2905599de6b0358dc485'
PINS = {
    'dpvo_probe_common.py':'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9',
    'dpvo_cpu_ops.py':'15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32',
    'manifest.json':'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375',
}
INPUT_NAMES = ('net','inp','corr','previous_indices','previous_valid','following_indices',
    'following_valid','kk_indices','kk_valid','kk_inverse','ij_indices','ij_valid','ij_inverse')
OUTPUT_NAMES = ('net_out','delta','weight')
DYNAMIC_AXES = {
    'net':{1:'E'},'inp':{1:'E'},'corr':{1:'E'},
    'previous_indices':{0:'E'},'previous_valid':{1:'E'},
    'following_indices':{0:'E'},'following_valid':{1:'E'},
    'kk_indices':{0:'Gkk',1:'Lkk'},'kk_valid':{0:'Gkk',1:'Lkk'},'kk_inverse':{0:'E'},
    'ij_indices':{0:'Gij',1:'Lij'},'ij_valid':{0:'Gij',1:'Lij'},'ij_inverse':{0:'E'},
    'net_out':{1:'E'},'delta':{1:'E'},'weight':{1:'E'},
}
EXPECTED_SHAPES = {
    'net':[1,'E',384],'inp':[1,'E',384],'corr':[1,'E',882],
    'previous_indices':['E'],'previous_valid':[1,'E',1],
    'following_indices':['E'],'following_valid':[1,'E',1],
    'kk_indices':['Gkk','Lkk'],'kk_valid':['Gkk','Lkk'],'kk_inverse':['E'],
    'ij_indices':['Gij','Lij'],'ij_valid':['Gij','Lij'],'ij_inverse':['E'],
    'net_out':[1,'E',384],'delta':[1,'E',2],'weight':[1,'E',2],
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda:file.read(1<<20),b''):
            h.update(data)
    return h.hexdigest()


def meta(array):
    return dict(shape=list(array.shape),dtype=str(array.dtype),bytes=array.nbytes,
        tensor_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def require_board():
    require(platform.system()=='Linux' and platform.machine()=='aarch64','RK3588 board-only model/export execution')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 device tree required')


def tensor_contract(value):
    tensor=value.type.tensor_type
    dims=[]
    for dim in tensor.shape.dim:
        if dim.HasField('dim_param'):
            require(bool(dim.dim_param),'Empty symbolic dimension')
            dims.append(dim.dim_param)
        else:
            require(dim.HasField('dim_value') and dim.dim_value>0,'Missing/invalid fixed dimension')
            dims.append(dim.dim_value)
    return dict(name=value.name,dtype=tensor.elem_type,shape=dims)


def check_contracts(model):
    inputs=[tensor_contract(v) for v in model.graph.input]
    outputs=[tensor_contract(v) for v in model.graph.output]
    require([v['name'] for v in inputs]==list(INPUT_NAMES),'All thirteen dynamic topology inputs must remain in order')
    require([v['name'] for v in outputs]==list(OUTPUT_NAMES),'Three Update outputs must remain in order')
    for value in inputs+outputs:
        name=value['name']
        dtype=7 if name.endswith('_indices') or name.endswith('_inverse') else 1
        require(value['dtype']==dtype and value['shape']==EXPECTED_SHAPES[name],
                'Dynamic axes/fixed B,C/dtype differ: '+name)
    return inputs,outputs


def check_rewrite(raw, final):
    prefix='v29_weight_sigmoid/'
    replaced=[n for n in raw.graph.node if n.op_type=='Sigmoid' and list(n.output)==['weight']]
    require(len(replaced)==1,'Exactly one original terminal weight Sigmoid required')
    old=replaced[0].SerializeToString()
    for name in ('initializer','input','output'):
        require([x.SerializeToString() for x in getattr(raw.graph,name)]==
            [x.SerializeToString() for x in getattr(final.graph,name)],'Sigmoid rewrite changed '+name)
    require([n.SerializeToString() for n in raw.graph.node if n.SerializeToString()!=old]==
        [n.SerializeToString() for n in final.graph.node if not n.name.startswith(prefix)],
        'Sigmoid rewrite changed nonterminal Update nodes')
    require(len(final.graph.node)==len(raw.graph.node)+8,'Unexpected stable sigmoid expansion')
    require([o.SerializeToString() for o in raw.opset_import]==[o.SerializeToString() for o in final.opset_import],
        'Rewrite changed ONNX opsets')


def export_graph(torch, wrapped, arguments, path):
    # Torch 2.3 uses the legacy exporter; newer synthetic environments may
    # default to dynamo, which uses different symbolic naming/shape semantics.
    options=dict(input_names=list(INPUT_NAMES),output_names=list(OUTPUT_NAMES),
        dynamic_axes=DYNAMIC_AXES,opset_version=17,do_constant_folding=True)
    if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
        options['dynamo']=False
    torch.onnx.export(wrapped,arguments,str(path),**options)


def compare(actual, expected):
    import numpy as np
    require(actual.dtype==expected.dtype==np.float32 and actual.shape==expected.shape and
        np.isfinite(actual).all() and np.isfinite(expected).all(),'Matching finite FP32 reference required')
    delta=actual.astype(np.float64)-expected.astype(np.float64)
    failed=int(np.count_nonzero(np.abs(delta)>1e-5+1e-4*np.abs(expected.astype(np.float64))))
    return dict(rtol=1e-4,atol=1e-5,failed_elements=failed,gate_passed=failed==0,
        bitwise_equal=actual.tobytes(order='C')==expected.tobytes(order='C'),max_abs=float(np.abs(delta).max()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('v26-source-dir','capture-dir','checkpoint','adapter','sigmoid-rewriter','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    # The guard precedes output creation, Torch imports, checkpoint loading and tracing.
    require_board()
    out=args.output_dir.resolve();require(out.is_relative_to(BASE),'New isolated board experiment output required')
    out.mkdir(exist_ok=False)
    started=time.monotonic();prior=os.sched_getaffinity(0);pins={};neural=None;before=update_before=None
    report=dict(schema='v31_DPVO_dynamic_Update_export',status='failed',script_sha256=sha(__file__),
        actual_board_execution=True,production_changed=False,original_VGGT_default_unchanged=True,
        different_model_from_VGGT=True,ORB_used=False,NPU_used=False,BA_executed=False,valid_pose_output=False,
        realtime_validated=False,models={},cleanup_errors=[],dynamic_axes=DYNAMIC_AXES,
        input_names=list(INPUT_NAMES),output_names=list(OUTPUT_NAMES),opset_version=17,
        ORT_numeric_gate_exercised=False,dynamic_shape_numeric_gate_exercised=False,
        RKNN_dynamic_shapes_support='not_tested',historical_sigmoid_ORT_receipt_sha256=HISTORICAL_SIGMOID_ORT_SHA)

    def pin(path, expected):
        path=Path(path).resolve();require(sha(path)==expected,'Pinned file differs: '+str(path))
        pins[str(path)]=expected
        return path

    def load_artifact(folder, item):
        require(Path(item['file']).name==item['file'],'Artifact basename required')
        path=pin(folder/item['file'],item['sha256']);require(path.stat().st_size==item['bytes'],'Artifact bytes differ')
        with np.load(path,allow_pickle=False) as saved:
            values={k:saved[k].copy() for k in saved.files}
        require({k:meta(v) for k,v in values.items()}==item['arrays'],'Original artifact raw metadata differs')
        return values

    def save(name, values):
        path=out/name;np.savez(path,**values)
        return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in values.items()})

    def model_record(path, model):
        inputs,outputs=check_contracts(model)
        return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),inputs=inputs,outputs=outputs,
            nodes=len(model.graph.node),initializers=len(model.graph.initializer),
            op_types=sorted({n.op_type for n in model.graph.node}),external_data_used=False)

    try:
        pin(__file__,report['script_sha256']);pin(args.adapter,ADAPTER_SHA);pin(args.sigmoid_rewriter,REWRITER_SHA)
        pin(args.checkpoint,CHECKPOINT_SHA)
        for name,digest in PINS.items():pin(args.v26_source_dir/name,digest)
        capture=json.loads(pin(args.capture_dir/'capture.json',CAPTURE_SHA).read_text())
        require(capture['status']=='captured_original_Update_states_not_VO' and capture['original_weights_unchanged'] is True
            and capture['sources_and_inputs_unchanged'] is True and capture['cleanup_errors']==[], 'Completed original capture required')
        cases=[c for c in capture['scenarios'] if c['name']=='first_motion_0_to_1_identity' and c['edges']==96]
        require(len(cases)==1 and cases[0]['states'][0]['step']==0,'Exact original E96 state0 example required')
        require({4,5,6,7}<=prior,'Four A76 CPUs required');os.sched_setaffinity(0,{4,5,6,7})
        import numpy as np
        import torch
        import onnx
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        common=module(args.v26_source_dir/'dpvo_probe_common.py','_v31_original_loader')
        adapter=module(args.adapter,'_v31_frozen_adapter');rewriter=module(args.sigmoid_rewriter,'_v31_frozen_sigmoid')
        require(tuple(adapter.INPUT_NAMES)==INPUT_NAMES and tuple(adapter.OUTPUT_NAMES)==OUTPUT_NAMES,'Pinned adapter interface differs')
        neural,loader=common.load_neural(args.v26_source_dir/'official',args.v26_source_dir/'manifest.json',
            args.checkpoint,CHECKPOINT_SHA,args.v26_source_dir/'dpvo_cpu_ops.py',cpu_ops_sha256=PINS['dpvo_cpu_ops.py'])
        for name,item in loader['source_hashes'].items():pin(args.v26_source_dir/'official'/name,item['sha256'])
        before={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
        require(len(before)==94 and before==capture['original_weights_before'],'Original complete 94 parameter tensors differ')
        update_before={k:meta(v.detach().numpy()) for k,v in neural.update.state_dict().items()}
        require(len(update_before)==50 and {'update.'+k:v for k,v in update_before.items()}==
            {k:v for k,v in before.items() if k.startswith('update.')},'Original Update parameter coverage differs')
        report.update(original_weights_before=before,original_Update_weights_before=update_before)
        wrapped=adapter.ExportableUpdate(neural.update).eval()
        state=cases[0]['states'][0]
        values=load_artifact(args.capture_dir,state['inputs']);golden=load_artifact(args.capture_dir,state['outputs'])
        topology=adapter.build_topology(*(torch.from_numpy(values[k]) for k in ('ii','jj','kk')))
        arguments=tuple(torch.from_numpy(values[k]) for k in ('net','ctx','corr'))+topology.tensors()
        adapter.validate_inputs(wrapped,arguments)
        input_values={k:v.detach().numpy().copy() for k,v in zip(INPUT_NAMES,arguments)}
        report.update(capture_sha256=CAPTURE_SHA,checkpoint_sha256=CHECKPOINT_SHA,adapter_sha256=ADAPTER_SHA,
            loader=loader,original_weights_before=before,original_Update_weights_before=update_before,
            cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,
            torch_version=torch.__version__,onnx_version=onnx.__version__,
            trace_example=dict(case=cases[0]['name'],step=0,edges=96,
                inputs_artifact=save('example-inputs.npz',input_values),
                reference_artifact=save('example-reference.npz',{'net_out':golden['net'],'delta':golden['delta'],'weight':golden['weight']}),
                graph_artifact=save('example-graph.npz',{k:values[k] for k in ('ii','jj','kk')}),
                topology_summary=topology.summary()))
        with torch.inference_mode():
            start=time.monotonic();result=wrapped(*arguments);elapsed=time.monotonic()-start
            checks={name:compare(v.detach().numpy(),golden[key]) for name,key,v in
                zip(OUTPUT_NAMES,('net','delta','weight'),result)}
            report.update(trace_example_wrapper_checks=checks,trace_example_wrapper_seconds=elapsed,
                trace_example_wrapper_gate_passed=all(c['gate_passed'] for c in checks.values()))
            require(report['trace_example_wrapper_gate_passed'],'Original FP32 trace-example wrapper gate failed')
            raw_path=out/'update-raw.onnx';start=time.monotonic()
            export_graph(torch,wrapped,arguments,raw_path)
            report['export_seconds']=time.monotonic()-start
            report['onnx_exporter']='legacy torch.onnx.export; dynamo=False when supported'
        raw=onnx.load(raw_path,load_external_data=False)
        require(all(t.data_location!=onnx.TensorProto.EXTERNAL and not t.external_data for t in raw.graph.initializer),
            'External ONNX initializer data prohibited')
        onnx.checker.check_model(raw,full_check=True)
        report['models']['raw']=model_record(raw_path,raw)
        start=time.monotonic();final=rewriter.rewrite(raw);check_rewrite(raw,final)
        onnx.checker.check_model(final,full_check=True);check_contracts(final)
        final_path=out/'update.onnx';onnx.save(final,final_path)
        report['rewrite_seconds']=time.monotonic()-start
        report['models']['final']=dict(model_record(final_path,final),source_onnx_sha256=report['models']['raw']['sha256'],
            role='dynamic_Update_terminal_stable_sigmoid')
        report['rewrite']=dict(source_sha256=REWRITER_SHA,terminal_weight_only=True,
            formula='Exp(Min(x,0))/(1+Exp(-Abs(x)))',original_initializers_byte_unchanged=True,
            other_nodes_byte_unchanged=True,IO_byte_unchanged=True)
        report.update(status='exported_dynamic_ONNX_numerical_validation_pending',limitations=[
            'Only original E96 example evaluated; other graph sizes require board numerical validation.',
            'Standard ONNX dynamic axes do not imply RKNN dynamic input support.',
            'Caller must build/validate actual topology for every graph; no pruning or topology constants.',
            'Final ONNX stable sigmoid has historical evidence but has not been executed in this export.',
            'No complete VO, fresh pose, accuracy or throughput claim.'])
    except Exception as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if neural is not None and before is not None:
            try:
                after={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()}
                update_after={k:meta(v.detach().numpy()) for k,v in neural.update.state_dict().items()}
                report.update(original_weights_after=after,original_Update_weights_after=update_after,
                    original_Update_weights_unchanged=(update_before is not None and update_after==update_before),
                    original_weights_unchanged=(after==before and (update_before is None or update_after==update_before)))
                require(report['original_weights_unchanged'],'Original parameters changed')
            except Exception as error:
                report['cleanup_errors'].append(repr(error));report['status']='failed'
        try:
            report['source_hashes_after']={path:sha(path) for path in pins}
            report['sources_and_inputs_unchanged']=report['source_hashes_after']==pins
            require(report['sources_and_inputs_unchanged'],'Pinned source/input changed')
        except Exception as error:
            report['cleanup_errors'].append(repr(error));report['status']='failed'
        try:
            os.sched_setaffinity(0,prior);report['affinity_restored']=set(os.sched_getaffinity(0))==set(prior)
            require(report['affinity_restored'],'Affinity restore failed')
        except Exception as error:
            report['cleanup_errors'].append(repr(error));report['status']='failed'
        report.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            driver_elapsed_seconds=time.monotonic()-started)
        (out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','trace_example_wrapper_gate_passed','cleanup_errors')}),flush=True)
    return 0 if report['status']=='exported_dynamic_ONNX_numerical_validation_pending' else 1


if __name__=='__main__':
    raise SystemExit(main())
