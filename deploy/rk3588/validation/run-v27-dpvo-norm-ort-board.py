#!/usr/bin/env python3
"""Board-only unchanged-weight CNN graph rewrite numerical gate, not VO."""
import argparse,hashlib,json,os,platform,resource,time,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2')
PARENT='527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82'
CAPTURE='e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088'
ORT='153da765b1145d7bb74bc635c1982ae74c652dac28cf99be56cb40534c44b667'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(a):
 return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def compare(a,b):
 import numpy as np
 require(a.dtype==b.dtype==np.float32 and a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Finite matching FP32 arrays required')
 x=a.astype(np.float64);y=b.astype(np.float64);delta=x-y
 n=int(np.count_nonzero(np.abs(delta)>1e-5+1e-4*np.abs(y)))
 return dict(rtol=1e-4,atol=1e-5,failed_elements=n,gate_passed=n==0,max_abs=float(np.abs(delta).max()),relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(y.ravel()),1e-30)),bitwise_equal=a.tobytes()==b.tobytes())
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('capture','export','old-ort','rewrite'):
  p.add_argument('--'+key+'-dir',type=Path,required=True);p.add_argument('--'+key+'-sha256',required=True)
 p.add_argument('--editor',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args(argv)
 require(platform.machine()=='aarch64' and 'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes().decode(),'RK3588 board only')
 out=a.output_dir.resolve();require(out.is_relative_to(BASE/'experiments'),'Isolated experiment output required');out.mkdir(exist_ok=False)
 pins={};prior=os.sched_getaffinity(0)
 report=dict(schema='v27_DPVO_InstanceNorm_rewrite_board_ORT',status='failed',script_sha256=sha(__file__),actual_board_execution=True,production_changed=False,ORB_used=False,valid_pose_output=False,complete_VO=False,realtime_validated=False,calls=[],cleanup_errors=[])
 def pin(path,digest):
  require(sha(path)==digest,'SHA mismatch: '+str(path));pins[str(path.resolve())]=digest;return path
 def artifact(directory,item):
  name=item['file'];require(Path(name).name==name,'Simple local artifact filename required');path=pin(directory/name,item['sha256']);require(path.stat().st_size==item['bytes'],'Artifact size mismatch');return path
 try:
  pin(Path(__file__),report['script_sha256'])
  c=json.loads(pin(a.capture_dir/'capture.json',a.capture_sha256).read_text())
  e=json.loads(pin(a.export_dir/'export.json',a.export_sha256).read_text())
  old=json.loads(pin(a.old_ort_dir/'ort.json',a.old_ort_sha256).read_text())
  rewrite=json.loads(pin(a.rewrite_dir/'rewrite.json',a.rewrite_sha256).read_text())
  require(a.capture_sha256==CAPTURE and a.old_ort_sha256==ORT and c['status']=='completed_encoder_cpu_probe_not_VO' and old['status']=='completed_CPU_ORT_gate' and old['exact_reference_gate_passed'],'Frozen original CPU reference/ORT gate required')
  require(e['onnx']['sha256']==PARENT==old['onnx']['sha256']==rewrite['input']['sha256'] and e['capture_sha256']==old['capture_sha256']==a.capture_sha256 and old['export_sha256']==a.export_sha256,'Original graph and reference chain mismatch')
  require(rewrite['status']=='rewritten_and_statically_checked_not_runtime_validated','Rewrite static gate required')
  for key in ('original_initializers_byte_exact','non_instance_norm_nodes_byte_exact','input_output_protobuf_byte_exact'):
   require(rewrite['preservation'][key] is True,'Graph changed beyond IN replacement: '+key)
  pin(a.editor,rewrite['script_sha256']);artifact(a.export_dir,e['onnx']);graph=artifact(a.rewrite_dir,rewrite['output'])
  require(len(rewrite['rewrites'])==10,'Exact 10 InstanceNorm replacements required')
  os.sched_setaffinity(0,{4,5,6,7})
  import numpy as np
  import onnxruntime as ort
  options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  begin=time.monotonic();session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
  report.update(init_seconds=time.monotonic()-begin,providers=session.get_providers(),ORT_version=ort.__version__,numpy_version=np.__version__,graph_optimization='ORT_DISABLE_ALL',cpu_affinity=sorted(os.sched_getaffinity(0)),intra_threads=4,inter_threads=1,capture_sha256=a.capture_sha256,export_sha256=a.export_sha256,old_ort_sha256=a.old_ort_sha256,rewrite_sha256=a.rewrite_sha256,onnx=rewrite['output'])
  require(report['providers']==['CPUExecutionProvider'],'CPU-only ORT required')
  for i in range(3):
   item=c['inputs'][i];x=np.load(artifact(a.capture_dir,item['input']),allow_pickle=False)
   require(meta(x)==item['array'] and x.shape==(1,1,3,464,608) and x.dtype==np.float32,'Original normalized input differs')
   ref=next(r for r in c['calls'] if r['frame_index']==i);o=old['pairs'][i]
   require(item['frame_id']==ref['frame_id'] and i==o['frame_index'],'Frame identity differs')
   with np.load(artifact(a.capture_dir,ref['artifact']),allow_pickle=False) as saved:expected={k:saved[k].copy() for k in ('fmap','imap')}
   with np.load(artifact(a.old_ort_dir,o['artifact']),allow_pickle=False) as saved:old_values={k:saved[k].copy() for k in ('fmap','imap')}
   for k in expected:require(meta(expected[k])==ref['arrays'][k] and meta(old_values[k])==o['arrays'][k],'Reference raw SHA differs')
   begin=time.monotonic();values=session.run(['fmap','imap'],{'normalized_bgr':x[:,0]});seconds=time.monotonic()-begin
   actual={k:v[:,None] for k,v in zip(('fmap','imap'),values)}
   checks={k:compare(actual[k],expected[k]) for k in actual};versus_old={k:compare(actual[k],old_values[k]) for k in actual}
   path=out/f'frame-{i}.npz';np.savez(path,**actual)
   report['calls'].append(dict(frame_index=i,frame_id=item['frame_id'],forward_seconds=seconds,versus_original_CPU=checks,versus_original_ORT=versus_old,arrays={k:meta(v) for k,v in actual.items()},artifact=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))))
   print(json.dumps(dict(frame=i,forward_seconds=seconds,passed=all(v['gate_passed'] for v in checks.values()))),flush=True)
  report['exact_FP32_gate_passed']=all(v['gate_passed'] for row in report['calls'] for key in ('versus_original_CPU','versus_original_ORT') for v in row[key].values())
  require(report['exact_FP32_gate_passed'],'Original FP32 gate failed; do not compile as validated')
  report['status']='completed_board_ORT_rewrite_gate_not_VO'
 except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
 finally:
  try:
   after={path:sha(path) for path in pins};report.update(source_hashes_after=after,sources_and_inputs_unchanged=after==pins);require(after==pins,'Pinned source changed')
  except Exception as exc:report['cleanup_errors'].append(repr(exc))
  try:os.sched_setaffinity(0,prior);report['affinity_restored']=os.sched_getaffinity(0)==prior;require(report['affinity_restored'],'Affinity restore failed')
  except Exception as exc:report['cleanup_errors'].append(repr(exc))
  report.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  if report['cleanup_errors']:report['status']='failed'
  (out/'ort.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report.get(k) for k in ('status','error','exact_FP32_gate_passed')}),flush=True)
 return 0 if report['status']=='completed_board_ORT_rewrite_gate_not_VO' else 1
if __name__=='__main__':raise SystemExit(main())
