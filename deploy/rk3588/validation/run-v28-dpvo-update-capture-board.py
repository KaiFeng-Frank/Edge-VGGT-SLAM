#!/usr/bin/env python3
"""Capture unchanged DPVO Update with real saved features, only on RK3588."""
import argparse,hashlib,importlib.util,json,os,platform,resource,time,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
PINS={'dpvo_probe_common.py':'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9',
      'dpvo_cpu_ops.py':'15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32',
      'manifest.json':'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375'}
CORR_SHA='0733a1544f2692af826fa22871a78cbafbe9fbfc1b828e53a6a1849a90c22804'
CHECKPOINT_SHA='30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metadata(a):return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('v26-source-dir','corr-dir','checkpoint','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv)
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board only')
    out=args.output_dir.resolve();require(out.is_relative_to(BASE),'Isolated board output required');out.mkdir(exist_ok=False)
    prior=os.sched_getaffinity(0);pins={};hooks=[];neural=None;before=None
    report=dict(schema='v28_DPVO_original_Update_capture',status='failed',script_sha256=sha(__file__),
        actual_board_execution=True,production_changed=False,ORB_used=False,NPU_used=False,
        BA_executed=False,valid_pose_output=False,realtime_validated=False,scenarios=[],cleanup_errors=[])
    def pin(path,expected):
        require(sha(path)==expected,'Pinned source differs: '+str(path));pins[str(path.resolve())]=expected;return path
    def artifact(folder,item,expected=None):
        import numpy as np
        name=item['file'];require(Path(name).name==name,'Artifact basename required')
        path=pin(folder/name,item['sha256']);require(path.stat().st_size==item['bytes'],'Artifact bytes differ')
        with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        expected=item['arrays'] if expected is None else expected
        require({k:metadata(v) for k,v in values.items()}==expected,'Artifact tensor metadata differs')
        return values
    def save(name,values):
        import numpy as np
        require(all(np.isfinite(v).all() for v in values.values()),'Nonfinite saved arrays')
        path=out/name;np.savez(path,**values)
        return dict(file=name,sha256=sha(path),bytes=path.stat().st_size,arrays={k:metadata(v) for k,v in values.items()})
    try:
        pin(Path(__file__),report['script_sha256'])
        for name,digest in PINS.items():pin(args.v26_source_dir/name,digest)
        pin(args.checkpoint,CHECKPOINT_SHA)
        corr=json.loads(pin(args.corr_dir/'run.json',CORR_SHA).read_text())
        require(corr['status']=='completed_native_corr_probe_not_VO' and corr['exact_FP32_gate_passed'] and len(corr['scenarios'])==4,'Verified v27 correlation evidence required')
        os.sched_setaffinity(0,{4,5,6,7})
        import numpy as np
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        common=load_module(args.v26_source_dir/'dpvo_probe_common.py','_v28_original_common')
        neural,loader=common.load_neural(args.v26_source_dir/'official',args.v26_source_dir/'manifest.json',args.checkpoint,CHECKPOINT_SHA,args.v26_source_dir/'dpvo_cpu_ops.py',cpu_ops_sha256=PINS['dpvo_cpu_ops.py'])
        before={k:metadata(v.detach().numpy()) for k,v in neural.state_dict().items()}
        require(before==corr['original_weights_before'],'Original inference parameters differ')
        report.update(loader=loader,original_weights_before=before,corr_receipt_sha256=CORR_SHA,cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,torch_version=torch.__version__,numpy_version=np.__version__)
        def execute(inputs):
            begin=time.monotonic()
            net,(delta,weight,unused)=neural.update(inputs['net'],inputs['ctx'],inputs['corr'],None,inputs['ii'],inputs['jj'],inputs['kk'])
            elapsed=time.monotonic()-begin
            require(unused is None,'Unexpected Update extra output')
            values={k:v.detach().numpy().copy() for k,v in dict(net=net,delta=delta,weight=weight).items()}
            require(all(np.isfinite(v).all() for v in values.values()) and ((values['weight']>=0)&(values['weight']<=1)).all(),'Invalid original Update outputs')
            return values,elapsed
        with torch.inference_mode():
            for old in corr['scenarios']:
                graph=artifact(args.corr_dir,old['graph'])
                row=next(c for c in old['calls'] if c['mode']=='reference' and c['iteration']==0)
                saved=artifact(args.corr_dir,row['artifact'])
                inputs={k:torch.from_numpy(graph['net0' if k=='net' else k]) for k in ('net','ctx','ii','jj','kk')}
                inputs['corr']=torch.from_numpy(saved['corr'])
                case=dict(name=old['name'],edges=old['edge_count'],calls=[],states=[],profile=None)
                report['scenarios'].append(case)
                golden=None
                for repetition in range(4):
                    values,elapsed=execute(inputs)
                    require(all(values[k].tobytes()==saved[k].tobytes() for k in values),'Fresh zero-state Update differs from v27')
                    case['calls'].append(dict(state_step=0,repetition=repetition,timing_role='cold' if repetition==0 else 'warm_same_state',seconds=elapsed,arrays={k:metadata(v) for k,v in values.items()}))
                    if repetition==0:
                        case['states'].append(dict(step=0,inputs=save(old['name']+'-state0-input.npz',{k:v.numpy() for k,v in inputs.items()}),outputs=save(old['name']+'-state0-output.npz',values)))
                        golden=values
                case['warm_mean_seconds']=sum(c['seconds'] for c in case['calls'][1:])/3
                current=golden
                for step in range(1,4):
                    recurrent=dict(inputs,net=torch.from_numpy(current['net']))
                    current,elapsed=execute(recurrent)
                    case['states'].append(dict(step=step,inputs=save(old['name']+f'-state{step}-input.npz',{k:v.numpy() for k,v in recurrent.items()}),outputs=save(old['name']+f'-state{step}-output.npz',current)))
                    case['calls'].append(dict(state_step=step,repetition=0,timing_role='recurrent_same_corr_single',seconds=elapsed,arrays={k:metadata(v) for k,v in current.items()}))
                starts={};spans={}
                def pre(name):
                    def hook(module,arguments):starts[name]=time.monotonic()
                    return hook
                def post(name):
                    def hook(module,arguments,result):spans.setdefault(name,[]).append(time.monotonic()-starts.pop(name))
                    return hook
                for name in ('corr','norm','c1','c2','agg_kk','agg_ij','gru','d','w'):
                    module=getattr(neural.update,name)
                    hooks.extend([module.register_forward_pre_hook(pre(name)),module.register_forward_hook(post(name))])
                profiled,elapsed=execute(inputs)
                for h in hooks:h.remove()
                hooks=[]
                require(all(profiled[k].tobytes()==golden[k].tobytes() for k in golden),'Instrumented output differs from original')
                require(all(len(v)==1 for v in spans.values()) and not starts,'Profile hooks unbalanced')
                case['profile']=dict(total_seconds=elapsed,nonoverlapping_stage_seconds={k:v[0] for k,v in spans.items()},other_and_hook_overhead_seconds=elapsed-sum(v[0] for v in spans.values()),bitwise_equal_uninstrumented=True,instrumented_not_in_warm_mean=True)
                print(json.dumps(dict(case=case['name'],warm_seconds=case['warm_mean_seconds'],profile=case['profile'])),flush=True)
        report.update(status='captured_original_Update_states_not_VO',original_weights_unchanged=before=={k:metadata(v.detach().numpy()) for k,v in neural.state_dict().items()},timing_scope='Uninstrumented original Update forward only; conversion of inputs performed before timing, owned NumPy output copies after timing. Topology/neighbor and SoftAgg CPU work inside original forward included. CNN/correlation/BA excluded.',limitations=['Fixed saved real features/correlation and four prescribed graphs; recurrence holds correlation/context/topology fixed and is not a tracked trajectory.','Profile uses hooks and a separate call; it is not added to the ordinary warm mean.','Short CPU measurement; clocks/cooling not controlled, no endurance or camera latency measurement.'])
        require(report['original_weights_unchanged'],'Original parameters changed')
    except Exception as e:report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        for h in hooks:
            try:h.remove()
            except Exception as e:report['cleanup_errors'].append(repr(e))
        try:
            report['source_hashes_after']={path:sha(path) for path in pins};report['sources_and_inputs_unchanged']=report['source_hashes_after']==pins
            require(report['sources_and_inputs_unchanged'],'Source or saved input changed')
            os.sched_setaffinity(0,prior);report['affinity_restored']=set(os.sched_getaffinity(0))==set(prior)
        except Exception as e:report['cleanup_errors'].append(repr(e));report['status']='failed'
        report.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'capture.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','cleanup_errors','peak_rss_mib')}),flush=True)
    return 0 if report['status']=='captured_original_Update_states_not_VO' else 1
if __name__=='__main__':raise SystemExit(main())
