#!/usr/bin/env python3
"""Board-only FP32 group reference checks and separate nonquantized compilation."""
import argparse, hashlib, json, os, platform, time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def artifact(root,row):
 path=(root/row['file']).resolve(strict=True)
 if not path.is_relative_to(root.resolve()) or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Artifact identity mismatch')
 return path
def write(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--mode',choices=['check','compile'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if platform.machine()!='aarch64' or b'rk3588' not in Path('/proc/device-tree/compatible').read_bytes():raise RuntimeError('RK3588 actual execution only')
 os.sched_setaffinity(0,{4,5,6,7})
 if sha(a.manifest)!=a.sha256:raise ValueError('Group manifest changed')
 m=json.loads(a.manifest.read_text());row=m['blocks'][0]
 assert m['schema']=='v58_original_VGGT_4_DINO_block_group_ONNX_manifest' and m['boundary_mode']=='fp32' and row['indices']==[0,1,2,3]
 assert m['original_parent_manifest_sha256']=='d68d05f984688e2c793f2e9d8320531202a4f9871790d7c2d96c74b5f41c9325'
 graph=artifact(a.manifest.parent,row);a.output.mkdir(parents=True,exist_ok=False)
 report=dict(status='running',mode=a.mode,manifest_sha256=a.sha256,graph_sha256=row['sha256'],script_sha256=sha(__file__),candidate_accepted=False,operations=[],checks=[],cleanup_errors=[])
 started=time.perf_counter();sdk=None
 try:
  if a.mode=='check':
   import numpy as np
   import onnxruntime as ort
   opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
   session=ort.InferenceSession(str(graph),sess_options=opts,providers=['CPUExecutionProvider'])
   parent_dir=Path('/mnt/vggt-data/vggt-slam2/experiments/vggt-v54-dino-graphs-r1')
   assert sha(parent_dir/'compiled.json')==m['original_parent_manifest_sha256']
   previous=[ort.InferenceSession(str(artifact(parent_dir,parent)),sess_options=opts,providers=['CPUExecutionProvider']) for parent in row['parents']]
   ref=Path(m['original_reference_capture_directory'])
   for tag in ['single0','single1']:
    refs=row['original_capture_references'][tag];x=np.load(artifact(ref,refs['input']),allow_pickle=False);y=np.load(artifact(ref,refs['output']),allow_pickle=False)
    assert x.shape==y.shape==(1,53,1024) and x.dtype==y.dtype==np.float32
    begin=time.perf_counter();actual=session.run(['output'],{'input':x})[0];elapsed=time.perf_counter()-begin
    assert actual.shape==y.shape and actual.dtype==y.dtype and np.isfinite(actual).all()
    prior=x.copy()
    for item in previous:prior=item.run(['output'],{'input':prior})[0]
    chain_equal=actual.tobytes()==prior.tobytes()
    delta=actual.astype(np.float64)-y.astype(np.float64);fail=np.abs(delta)>(1e-5+1e-4*np.abs(y.astype(np.float64)))
    np.save(a.output/(tag+'.npy'),actual,allow_pickle=False)
    report['checks'].append(dict(input=tag,strict_rtol=1e-4,strict_atol=1e-5,strict_pass=not bool(fail.any()),failed_elements=int(fail.sum()),max_abs=float(np.abs(delta).max()),relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(y.astype(np.float64))),ORT_CPU_seconds=elapsed,byte_equal_to_four_separate_FP32_ONNX=chain_equal,versus_separate_max_abs=float(np.abs(actual.astype(np.float64)-prior.astype(np.float64)).max())))
   report['status']='checked_original_FP32_references'
   report['all_original_FP32_checks_passed']=all(v['strict_pass'] for v in report['checks'])
   report['all_separate_ONNX_checks_byte_equal']=all(v['byte_equal_to_four_separate_FP32_ONNX'] for v in report['checks'])
  else:
   prior=json.loads((a.manifest.parent/'ort-check-r2/result.json').read_text())
   assert prior['manifest_sha256']==a.sha256 and prior['all_separate_ONNX_checks_byte_equal'] and len(prior['checks'])==2
   report['original_FP32_gate_passed']=prior['all_original_FP32_checks_passed']
   report['compile_scope']='diagnostic of an unchanged exported graph chain; original Torch FP32 failures retained, no numerical acceptance'
   report['check_receipt_sha256']=sha(a.manifest.parent/'ort-check-r2/result.json')
   from rknn.api import RKNN
   sdk=RKNN(verbose=False)
   cfg=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,mean_values=None,std_values=None,quant_img_RGB2BGR=False,model_pruning=False,remove_reshape=False,quantize_weight=False,enable_flash_attention=False)
   report['configuration']=cfg
   os.chdir(a.output);os.environ['TMPDIR']=str(a.output)
   for name,kwargs in [('config',cfg),('load_onnx',{'model':str(graph)}),('build',{'do_quantization':False}),('export_rknn',{'export_path':str(a.output/'group.rknn')})]:
    begin=time.perf_counter();code=getattr(sdk,name)(**kwargs);report['operations'].append(dict(name=name,code=code,seconds=time.perf_counter()-begin));write(a.output/'result.json',report)
    if code:raise RuntimeError(name+' failed: '+str(code))
   report['rknn']=dict(file='group.rknn',bytes=(a.output/'group.rknn').stat().st_size,sha256=sha(a.output/'group.rknn'))
   report['status']='compiled_not_quality_accepted'
 except BaseException as exc:
  report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
 finally:
  if sdk is not None:
   try:sdk.release()
   except BaseException as exc:report['cleanup_errors'].append(str(exc));report['status']='failed_cleanup'
  report['seconds']=time.perf_counter()-started;write(a.output/'result.json',report)
 print(json.dumps({k:report.get(k) for k in ['status','all_original_FP32_checks_passed','checks','seconds','rknn']}),flush=True)
if __name__=='__main__':main()
