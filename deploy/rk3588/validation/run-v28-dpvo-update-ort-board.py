#!/usr/bin/env python3
"""Board CPU ORT full Update with captured nonzero states/runtime topology."""
import argparse,hashlib,json,os,platform,resource,time,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
CAPTURE_SHA='87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(a):return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def compare(a,b):
    import numpy as np
    require(a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Matching finite FP32 required')
    d=a.astype(np.float64)-b.astype(np.float64);n=int(np.count_nonzero(np.abs(d)>1e-5+1e-4*np.abs(b.astype(np.float64))))
    return dict(rtol=1e-4,atol=1e-5,failed_elements=n,gate_passed=n==0,max_abs=float(np.abs(d).max()),relative_l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(b.astype(np.float64).ravel()),1e-30)),bitwise_equal=a.tobytes()==b.tobytes())
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture-dir','export-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--export-sha256',required=True);a=p.parse_args(argv)
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board only')
    out=a.output_dir.resolve();require(out.is_relative_to(BASE),'Isolated output required');out.mkdir(exist_ok=False)
    prior=os.sched_getaffinity(0);pins={};sessions={}
    r=dict(schema='v28_DPVO_full_Update_board_ORT',status='failed',script_sha256=sha(__file__),actual_board_execution=True,
        production_changed=False,ORB_used=False,BA_executed=False,valid_pose_output=False,realtime_validated=False,modes=[],cleanup_errors=[])
    def pin(path,expected):
        require(sha(path)==expected,'Pinned file differs: '+str(path));pins[str(path.resolve())]=expected;return path
    def artifact(folder,item):
        name=item['file'];require(Path(name).name==name,'Artifact basename required')
        path=pin(folder/name,item['sha256']);require(path.stat().st_size==item['bytes'],'Artifact bytes differ')
        with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        require({k:meta(v) for k,v in values.items()}==item['arrays'],'Saved tensor metadata differs');return values
    def save(name,values):
        path=out/name;np.savez(path,**values);return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in values.items()})
    try:
        pin(Path(__file__),r['script_sha256'])
        capture=json.loads(pin(a.capture_dir/'capture.json',CAPTURE_SHA).read_text())
        exported=json.loads(pin(a.export_dir/'export.json',a.export_sha256).read_text())
        require(capture['status']=='captured_original_Update_states_not_VO' and capture['original_weights_unchanged'] and exported['status']=='exported_Update_board_ORT_pending' and exported['capture_sha256']==CAPTURE_SHA and exported['original_weights_unchanged'],'Verified original/export chain required')
        os.sched_setaffinity(0,{4,5,6,7})
        import numpy as np
        import onnxruntime as ort
        r.update(capture_sha256=CAPTURE_SHA,export_sha256=a.export_sha256,wrapper_FP32_gate_passed=exported['exact_wrapper_FP32_gate_passed'],ORT_version=ort.__version__,numpy_version=np.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),intra_threads=4,inter_threads=1)
        for mode,level in [('disabled',ort.GraphOptimizationLevel.ORT_DISABLE_ALL),('enabled',ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]:
            record=dict(name=mode,sessions={},scenarios=[]);r['modes'].append(record)
            for key,model in exported['models'].items():
                path=pin(a.export_dir/model['file'],model['sha256']);require(path.stat().st_size==model['bytes'],'Model bytes differ')
                opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;opts.graph_optimization_level=level
                begin=time.monotonic();session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']);elapsed=time.monotonic()-begin
                require(session.get_providers()==['CPUExecutionProvider'],'CPU provider required')
                require([x.name for x in session.get_inputs()]==exported['input_names'],'Runtime explicit input names differ')
                sessions[key]=session;record['sessions'][key]=dict(init_seconds=elapsed,model_sha256=model['sha256'],providers=session.get_providers(),inputs=[dict(name=x.name,shape=x.shape,type=x.type) for x in session.get_inputs()])
            for case in capture['scenarios']:
                rewritten=next(x for x in exported['scenarios'] if x['name']==case['name']);session=sessions[rewritten['model_key']]
                topo=artifact(a.export_dir,rewritten['topology']);row=dict(name=case['name'],edges=case['edges'],model_key=rewritten['model_key'],calls=[]);record['scenarios'].append(row)
                for state in case['states']:
                    inputs=artifact(a.capture_dir,state['inputs']);golden=artifact(a.capture_dir,state['outputs'])
                    wrapped=artifact(a.export_dir,next(x for x in rewritten['states'] if x['step']==state['step'])['outputs'])
                    begin=time.monotonic();feed=dict(net=np.ascontiguousarray(inputs['net']),inp=np.ascontiguousarray(inputs['ctx']),corr=np.ascontiguousarray(inputs['corr']),**{k:np.ascontiguousarray(v) for k,v in topo.items()});pack=time.monotonic()-begin
                    require(set(feed)==set(exported['input_names']),'Explicit feed input set differs')
                    first_metadata=None
                    for repeat in range(4 if state['step']==0 else 1):
                        begin=time.monotonic();result=session.run(exported['output_names'],feed);elapsed=time.monotonic()-begin
                        values=dict(zip(('net','delta','weight'),result));metadata={k:meta(v) for k,v in values.items()}
                        checks={k:compare(v,golden[k]) for k,v in values.items()};wrapper_checks={k:compare(v,wrapped[k]) for k,v in values.items()}
                        if first_metadata is None:first_metadata=metadata
                        require(metadata==first_metadata,'Same ORT input repeated differently')
                        call=dict(step=state['step'],repetition=repeat,timing_role='cold_case' if state['step']==0 and repeat==0 else 'warm_same_state' if state['step']==0 else 'nonzero_captured_state_single',forward_seconds=elapsed,feed_prepare_seconds=pack,versus_original=checks,versus_wrapper=wrapper_checks,arrays=metadata,repeat_bitwise_equal=True)
                        if repeat==0:call['outputs']=save(mode+'-'+case['name']+f"-state{state['step']}.npz",values)
                        row['calls'].append(call)
                row['warm_mean_seconds']=sum(c['forward_seconds'] for c in row['calls'] if c['timing_role']=='warm_same_state')/3
                print(json.dumps(dict(mode=mode,case=case['name'],warm_mean_seconds=row['warm_mean_seconds'],failed_elements=sum(x['failed_elements'] for c in row['calls'] for x in c['versus_original'].values()))),flush=True)
            sessions.clear();del session
            record['exact_original_FP32_gate_passed']=all(x['gate_passed'] for s in record['scenarios'] for c in s['calls'] for x in c['versus_original'].values())
            record['exact_wrapper_FP32_gate_passed']=all(x['gate_passed'] for s in record['scenarios'] for c in s['calls'] for x in c['versus_wrapper'].values())
        r.update(status='completed_Update_ORT_probe_not_VO',exact_original_FP32_gate_passed=all(m['exact_original_FP32_gate_passed'] for m in r['modes']),timing_scope='session.run through owned output arrays, includes runtime input handling/output allocation; feed construction and graph topology preparation separately reported/excluded. File I/O, metadata/checks and initialization excluded. Four repeats only for state0; other nonzero states each single call.',limitations=['Captured recurrence states are fed independently from original CPU references; these calls do not validate accumulated ORT feedback drift.','Graph capacities fixed; topology values supplied as runtime inputs.','No CNN/correlation/BA/pose or sustained end-to-end timing.','Numerical failures remain diagnostic; completion of execution is not FP32 acceptance.'])
    except Exception as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        sessions.clear()
        try:
            r['source_hashes_after']={path:sha(path) for path in pins};r['sources_and_inputs_unchanged']=r['source_hashes_after']==pins;require(r['sources_and_inputs_unchanged'],'Source/input changed')
            os.sched_setaffinity(0,prior);r['affinity_restored']=set(os.sched_getaffinity(0))==set(prior)
        except Exception as e:r['cleanup_errors'].append(repr(e));r['status']='failed'
        r.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (out/'ort.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('status','error','exact_original_FP32_gate_passed','cleanup_errors','peak_rss_mib')}),flush=True)
    return 0 if r['status']=='completed_Update_ORT_probe_not_VO' else 1
if __name__=='__main__':raise SystemExit(main())
