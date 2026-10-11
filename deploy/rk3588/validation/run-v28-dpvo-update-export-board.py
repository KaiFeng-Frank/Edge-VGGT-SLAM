#!/usr/bin/env python3
"""Board-only whole DPVO Update export with explicit runtime topology inputs."""
import argparse,hashlib,importlib.util,json,os,platform,resource,time,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
CAPTURE_SHA='87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
CHECKPOINT_SHA='30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
PINS={'dpvo_probe_common.py':'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9','dpvo_cpu_ops.py':'15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32','manifest.json':'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375'}
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def meta(a):return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def module(path,name):
    import sys
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def compare(a,b):
    import numpy as np
    require(a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Matching finite FP32 arrays required')
    d=a.astype(np.float64)-b.astype(np.float64);n=int(np.count_nonzero(np.abs(d)>1e-5+1e-4*np.abs(b.astype(np.float64))))
    return dict(rtol=1e-4,atol=1e-5,failed_elements=n,gate_passed=n==0,max_abs=float(np.abs(d).max()),relative_l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(b.astype(np.float64).ravel()),1e-30)),bitwise_equal=a.tobytes()==b.tobytes())
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('v26-source-dir','capture-dir','checkpoint','adapter','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--adapter-sha256',required=True);a=p.parse_args(argv)
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board only')
    out=a.output_dir.resolve();require(out.is_relative_to(BASE),'Isolated output required');out.mkdir(exist_ok=False)
    prior=os.sched_getaffinity(0);pins={};neural=None;before=None
    r=dict(schema='v28_DPVO_full_Update_export',status='failed',script_sha256=sha(__file__),actual_board_execution=True,
        production_changed=False,ORB_used=False,BA_executed=False,valid_pose_output=False,realtime_validated=False,models={},scenarios=[],cleanup_errors=[])
    def pin(path,expected):
        require(sha(path)==expected,'Pinned file differs: '+str(path));pins[str(path.resolve())]=expected;return path
    def artifact(folder,item):
        name=item['file'];require(Path(name).name==name,'Artifact basename required')
        path=pin(folder/name,item['sha256']);require(path.stat().st_size==item['bytes'],'Artifact bytes differ')
        with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        require({k:meta(v) for k,v in values.items()}==item['arrays'],'Artifact metadata differs');return values
    def save(name,values):
        path=out/name;np.savez(path,**values);return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in values.items()})
    try:
        pin(Path(__file__),r['script_sha256']);pin(a.adapter,a.adapter_sha256);pin(a.checkpoint,CHECKPOINT_SHA)
        for name,digest in PINS.items():pin(a.v26_source_dir/name,digest)
        capture=json.loads(pin(a.capture_dir/'capture.json',CAPTURE_SHA).read_text())
        require(capture['status']=='captured_original_Update_states_not_VO' and capture['original_weights_unchanged'] and capture['sources_and_inputs_unchanged'],'Verified original Update capture required')
        os.sched_setaffinity(0,{4,5,6,7})
        import numpy as np
        import torch
        import onnx
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        common=module(a.v26_source_dir/'dpvo_probe_common.py','_v28_export_common');adapter=module(a.adapter,'_v28_update_adapter')
        neural,loader=common.load_neural(a.v26_source_dir/'official',a.v26_source_dir/'manifest.json',a.checkpoint,CHECKPOINT_SHA,a.v26_source_dir/'dpvo_cpu_ops.py',cpu_ops_sha256=PINS['dpvo_cpu_ops.py'])
        before={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()};require(before==capture['original_weights_before'],'Original parameters differ')
        wrapped=adapter.ExportableUpdate(neural.update).eval()
        r.update(loader=loader,capture_sha256=CAPTURE_SHA,adapter_sha256=a.adapter_sha256,input_names=list(adapter.INPUT_NAMES),output_names=['net_out','delta','weight'],cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,torch_version=torch.__version__,onnx_version=onnx.__version__,original_weights_before=before)
        with torch.inference_mode():
            for case in capture['scenarios']:
                row=dict(name=case['name'],edges=case['edges'],states=[]);r['scenarios'].append(row)
                first=artifact(a.capture_dir,case['states'][0]['inputs'])
                begin=time.monotonic();topo=adapter.build_topology(*(torch.from_numpy(first[k]) for k in ('ii','jj','kk')));row['topology_prepare_seconds']=time.monotonic()-begin
                full_topology=topo.numpy_dict();topology={k:full_topology[k] for k in adapter.INPUT_NAMES[3:]}
                row['topology_summary']=topo.summary()
                row['topology']=save(case['name']+'-topology.npz',topology)
                row['topology_bytes']=sum(v.nbytes for v in topology.values())
                export_args=None
                for state in case['states']:
                    inputs=artifact(a.capture_dir,state['inputs']);golden=artifact(a.capture_dir,state['outputs'])
                    require(all(inputs[k].tobytes()==first[k].tobytes() for k in ('ii','jj','kk')),'Graph changed within captured recurrence')
                    args=tuple(torch.from_numpy(inputs[k]) for k in ('net','ctx','corr'))+topo.tensors()
                    adapter.validate_inputs(wrapped,args)
                    begin=time.monotonic();values=wrapped(*args);elapsed=time.monotonic()-begin
                    outputs={k:v.detach().numpy().copy() for k,v in zip(('net','delta','weight'),values)}
                    checks={k:compare(v,golden[k]) for k,v in outputs.items()}
                    row['states'].append(dict(step=state['step'],forward_seconds=elapsed,checks=checks,outputs=save(case['name']+f"-state{state['step']}-wrapped.npz",outputs)))
                    if export_args is None:export_args=args
                modelkey='E'+str(case['edges']);row['model_key']=modelkey
                if modelkey not in r['models']:
                    path=out/(modelkey+'.onnx');begin=time.monotonic()
                    torch.onnx.export(wrapped,export_args,str(path),input_names=list(adapter.INPUT_NAMES),output_names=r['output_names'],opset_version=17,do_constant_folding=True)
                    elapsed=time.monotonic()-begin;model=onnx.load(str(path));onnx.checker.check_model(model)
                    actual_names=[x.name for x in model.graph.input]
                    require(actual_names==r['input_names'],'Exporter pruned or reordered explicit topology inputs')
                    def contract(x):return dict(name=x.name,dtype=x.type.tensor_type.elem_type,shape=[d.dim_value for d in x.type.tensor_type.shape.dim])
                    r['models'][modelkey]=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),export_seconds=elapsed,inputs=[contract(x) for x in model.graph.input],outputs=[contract(x) for x in model.graph.output],nodes=len(model.graph.node),initializers=len(model.graph.initializer),op_types=sorted({x.op_type for x in model.graph.node}),runtime_topology_inputs_preserved=True)
                print(json.dumps(dict(case=case['name'],topology_seconds=row['topology_prepare_seconds'],gates=[{k:x['gate_passed'] for k,x in s['checks'].items()} for s in row['states']])),flush=True)
        r['exact_wrapper_FP32_gate_passed']=all(x['gate_passed'] for s in r['scenarios'] for c in s['states'] for x in c['checks'].values())
        r['original_weights_unchanged']=before=={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()};require(r['original_weights_unchanged'],'Original parameters changed')
        r.update(status='exported_Update_board_ORT_pending',numeric_validation='wrapper_passed' if r['exact_wrapper_FP32_gate_passed'] else 'wrapper_failed_preserved_as_diagnostic',timing_scope='Whole wrapper forward; topology preparation separately measured, output copies and validation excluded. These are single calls, not steady-state throughput.',limitations=['Fixed E/group capacities, topology values remain runtime inputs.','No BA or tracked trajectory.','Export/checker success is not numerical acceptance; original FP32 failures remain visible.'])
    except Exception as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        try:
            r['source_hashes_after']={path:sha(path) for path in pins};r['sources_and_inputs_unchanged']=r['source_hashes_after']==pins;require(r['sources_and_inputs_unchanged'],'Source/input changed')
            os.sched_setaffinity(0,prior);r['affinity_restored']=set(os.sched_getaffinity(0))==set(prior)
        except Exception as e:r['cleanup_errors'].append(repr(e));r['status']='failed'
        r.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'export.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('status','error','exact_wrapper_FP32_gate_passed','cleanup_errors','peak_rss_mib')}),flush=True)
    return 0 if r['status']=='exported_Update_board_ORT_pending' else 1
if __name__=='__main__':raise SystemExit(main())
