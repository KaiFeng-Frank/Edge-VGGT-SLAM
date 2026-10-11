#!/usr/bin/env python3
"""Board-only reference/native correlation + unchanged Update comparison."""
import argparse,hashlib,importlib.util,json,os,platform,resource,time,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2')
PINS={'dpvo_cpu_ops.py':'15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32','dpvo_probe_common.py':'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9','manifest.json':'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375','run-v26-dpvo-motion-board.py':'d266bdd513faf7c180688ec6724a3219afd81b8782b6c5b48c46c46164e833ae'}
CAPTURE='e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088'
MOTION='55a7db19d1a9bfbf986da62ed999fc8b122add0fca84f01681e6ad0b82ba4e86'
CHECKPOINT='30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(a):return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def comparison(a,b):
 import numpy as np
 require(a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Finite matching FP32 required')
 d=a.astype(np.float64)-b.astype(np.float64);n=int(np.count_nonzero(np.abs(d)>1e-5+1e-4*np.abs(b.astype(np.float64))))
 return dict(rtol=1e-4,atol=1e-5,failed_elements=n,gate_passed=n==0,max_abs=float(np.abs(d).max()),relative_l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(b.astype(np.float64).ravel()),1e-30)),bitwise_equal=a.tobytes()==b.tobytes())
def shifted_coords(original):
 import numpy as np
 out=original.copy();e=np.arange(out.shape[1],dtype=np.int64)
 out[0,:,0]+=(np.float32(.17)*(e%7-3).astype(np.float32))[:,None,None]
 out[0,:,1]+=(np.float32(.23)*(e%5-2).astype(np.float32))[:,None,None]
 return out

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('v26-source-dir','capture-dir','motion-dir','checkpoint','native-source-dir','library','build-receipt','output-dir'):
  p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--build-sha256',required=True)
 a=p.parse_args(argv)
 require(platform.machine()=='aarch64' and 'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes().decode(),'RK3588 board only')
 out=a.output_dir.resolve();require(out.is_relative_to(BASE/'experiments'),'Isolated output required');out.mkdir(exist_ok=False)
 prior=os.sched_getaffinity(0);pins={};neural=None;state=None;cached=None
 report=dict(schema='v27_DPVO_native_corr_board_probe',status='failed',script_sha256=sha(__file__),actual_board_execution=True,production_changed=False,ORB_used=False,NPU_used=False,BA_executed=False,valid_pose_output=False,tracked_trajectory=False,realtime_validated=False,scenarios=[],cleanup_errors=[])
 def pin(path,expected):
  require(sha(path)==expected,'Pinned file differs: '+str(path));pins[str(path.resolve())]=expected;return path
 def load_artifact(folder,item,metadata):
  name=item['file'];require(Path(name).name==name,'Safe artifact basename required');path=pin(folder/name,item['sha256']);require(path.stat().st_size==item['bytes'],'Artifact size differs')
  with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
  require(set(values)==set(metadata) and all(meta(v)==metadata[k] and np.isfinite(v).all() for k,v in values.items()),'Artifact array metadata/finite mismatch')
  return values
 def save(filename,values):
  path=out/filename;np.savez(path,**values)
  return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in values.items()})
 try:
  pin(Path(__file__),report['script_sha256'])
  build=json.loads(pin(a.build_receipt,a.build_sha256).read_text());require(build['status']=='compiled_and_synthetic_smoke_passed','Native build must be verified')
  require(build['output']['sha256']==sha(a.library),'Library differs from build')
  pin(a.library,build['output']['sha256'])
  for name,digest in build['source_sha256'].items():pin(a.native_source_dir/name,digest)
  for name,digest in PINS.items():pin(a.v26_source_dir/name,digest)
  pin(a.checkpoint,CHECKPOINT)
  c=json.loads(pin(a.capture_dir/'capture.json',CAPTURE).read_text());m=json.loads(pin(a.motion_dir/'motion.json',MOTION).read_text())
  require(c['status']=='completed_encoder_cpu_probe_not_VO' and m['status']=='completed_dynamic_neural_probe_not_VO' and m['capture_sha256']==CAPTURE,'Original CPU reference chain required')
  os.sched_setaffinity(0,{4,5,6,7})
  import numpy as np
  import torch
  torch.set_num_threads(4);torch.set_num_interop_threads(1)
  report.update(cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=4,torch_interop_threads=1,torch_version=torch.__version__,numpy_version=np.__version__,capture_sha256=CAPTURE,motion_sha256=MOTION,build_sha256=a.build_sha256,library_sha256=sha(a.library),OMP_WAIT_POLICY=os.environ.get('OMP_WAIT_POLICY'),GOMP_SPINCOUNT=os.environ.get('GOMP_SPINCOUNT'))
  ops=module(a.v26_source_dir/'dpvo_cpu_ops.py','_v27_frozen_ops');common=module(a.v26_source_dir/'dpvo_probe_common.py','_v27_frozen_common');native=module(a.native_source_dir/'dpvo_corr.py','_v27_native')
  neural,loader=common.load_neural(a.v26_source_dir/'official',a.v26_source_dir/'manifest.json',a.checkpoint,CHECKPOINT,a.v26_source_dir/'dpvo_cpu_ops.py',cpu_ops_sha256=PINS['dpvo_cpu_ops.py'])
  state={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()};require(state==c['state_before']==m['original_weights_before'],'Original 94 parameter bytes differ')
  report.update(loader=loader,original_weights_before=state)
  patch=load_artifact(a.motion_dir,m['patches'],m['patches']['arrays'])
  values=[]
  for i in range(3):
   row=next(x for x in c['calls'] if x['frame_index']==i);values.append(load_artifact(a.capture_dir,row['artifact'],row['arrays'])['fmap'])
  fmap=np.ascontiguousarray(np.concatenate(values,axis=1));gmap=patch['gmap'];feature_key=meta(fmap)['tensor_sha256']+':'+meta(gmap)['tensor_sha256']
  report['features']=dict(fmap=meta(fmap),gmap=meta(gmap),cache_key=feature_key,frame_ids=m['frame_ids'])
  f_tensor=torch.from_numpy(fmap);g_tensor=torch.from_numpy(gmap)
  with torch.inference_mode():
   for parent in m['scenarios']:
    original=load_artifact(a.motion_dir,parent['graph'],parent['graph']['arrays'])
    oldref=load_artifact(a.motion_dir,parent['calls'][0]['artifact'],parent['calls'][0]['artifact']['arrays'])
    for shifted in (False,True):
     graph={k:v.copy() for k,v in original.items()}
     if shifted:graph['coords']=shifted_coords(graph['coords'])
     name=parent['name']+('_shifted' if shifted else '_identity')
     case=dict(name=name,parent=parent['name'],shifted=shifted,edge_count=int(graph['kk'].size),graph=save(name+'-graph.npz',graph),calls=[],fresh_zero_net_each_call=True,feedback_iterations=0)
     report['scenarios'].append(case)
     tg={k:torch.from_numpy(v) for k,v in graph.items()};reference=None;mode_hashes={}
     cached=native.PreparedCorrelation(a.library,gmap,fmap,threads=4,cache_key=feature_key,library_sha256=sha(a.library))
     case['cached_preparation']=cached.preparation
     for iteration in range(4):
      for mode in ('reference','native_one_shot','native_cached'):
       begin=time.monotonic()
       if mode=='reference':
        corr=ops.corr_two_levels(g_tensor,f_tensor,tg['coords'],tg['kk'],tg['jj'],radius=3).numpy();timing=None
       elif mode=='native_one_shot':
        corr,timing=native.correlate_once(a.library,gmap,fmap,graph['coords'],graph['kk'],graph['jj'],threads=4,cache_key=feature_key,library_sha256=report['library_sha256'])
       else:corr,timing=cached.correlate(graph['coords'],graph['kk'],graph['jj'],cache_key=feature_key,radius=3)
       after_corr=time.monotonic()
       net,(delta,weight,unused)=neural.update(tg['net0'],tg['ctx'],torch.from_numpy(corr),None,tg['ii'],tg['jj'],tg['kk'])
       after_update=time.monotonic();require(unused is None,'Original Update output differs')
       outputs=dict(corr=corr,net=net.numpy(),delta=delta.numpy(),weight=weight.numpy())
       hashes={k:meta(v) for k,v in outputs.items()}
       require(all(np.isfinite(v).all() for v in outputs.values()) and bool(((outputs['weight']>=0)&(outputs['weight']<=1)).all()),'Nonfinite or invalid Update outputs')
       if mode=='reference' and reference is None:reference={k:v.copy() for k,v in outputs.items()}
       checks={k:comparison(v,reference[k]) for k,v in outputs.items()}
       row=dict(mode=mode,iteration=iteration,timing_role='cold' if iteration==0 else 'warm',corr_seconds=after_corr-begin,update_seconds=after_update-after_corr,total_dynamic_seconds=after_update-begin,native_timing=timing,arrays=hashes,versus_reference=checks)
       if not shifted and mode=='reference':
        row['versus_v26_saved']={k:comparison(v,oldref[k]) for k,v in outputs.items()};require(all(x['gate_passed'] for x in row['versus_v26_saved'].values()),'Fresh original reference differs from v26')
       if mode not in mode_hashes:mode_hashes[mode]=hashes;row['artifact']=save(name+'-'+mode+'.npz',outputs)
       row['repeat_bitwise_equal']=hashes==mode_hashes[mode];require(row['repeat_bitwise_equal'],'Fixed graph output changed across repetitions')
       case['calls'].append(row)
       print(json.dumps(dict(scenario=name,mode=mode,iteration=iteration,corr_seconds=row['corr_seconds'],update_seconds=row['update_seconds'],gate=all(x['gate_passed'] for x in checks.values()))),flush=True)
     case['cached_metrics_before_close']=cached.metrics();t=time.monotonic();cached.close();case['cached_close_seconds']=time.monotonic()-t;case['cached_metrics_after_close']=cached.metrics();cached=None
     case['warm_means']={mode:{key:sum(r[key] for r in case['calls'] if r['mode']==mode and r['iteration']>0)/3 for key in ('corr_seconds','update_seconds','total_dynamic_seconds')} for mode in ('reference','native_one_shot','native_cached')}
     require(all(meta(v)==case['graph']['arrays'][k] for k,v in graph.items()),'Update mutated graph/context/net0')
  report['exact_FP32_gate_passed']=all(v['gate_passed'] for s in report['scenarios'] for r in s['calls'] for v in r['versus_reference'].values())
  report.update(status='completed_native_corr_probe_not_VO',timing_scope='Interleaved reference/one-shot/cached on every repetition. corr includes each wrapper call; one-shot includes prepare+query+close. Cached setup is separately reported and excluded from query time. Update includes NumPy-to-Torch wrapper and entire unchanged original Update. CNN/patch/graph setup and BA excluded.',limitations=['Fixed three captured views; no streamed frames, camera poses, BA or VO lifecycle.','CPU reference recomputes pyramids each call; cached mode reuses owned immutable features under explicit version key.','Short interleaved measurements without clock/governor/cooling changes, not sustained or end-to-end latency.','All original weights retained; only correlation implementation differs. FP32 gate remains 1e-4/1e-5.'])
 except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
 finally:
  if cached is not None:
   try:cached.close()
   except Exception as exc:report['cleanup_errors'].append(repr(exc))
  try:
   report['original_weights_unchanged']=neural is not None and state=={k:meta(v.detach().numpy()) for k,v in neural.state_dict().items()};require(report['original_weights_unchanged'],'Original weights unavailable or changed')
   after={path:sha(path) for path in pins};report.update(source_hashes_after=after,sources_and_inputs_unchanged=after==pins);require(after==pins,'Pinned source changed')
  except Exception as exc:report['cleanup_errors'].append(repr(exc))
  try:os.sched_setaffinity(0,prior);report['affinity_restored']=os.sched_getaffinity(0)==prior;require(report['affinity_restored'],'Affinity restore failed')
  except Exception as exc:report['cleanup_errors'].append(repr(exc))
  if report['cleanup_errors']:report['status']='failed'
  report.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  (out/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report.get(k) for k in ('status','error','exact_FP32_gate_passed','cleanup_errors')}),flush=True)
 return 0 if report['status']=='completed_native_corr_probe_not_VO' else 1
if __name__=='__main__':raise SystemExit(main())
