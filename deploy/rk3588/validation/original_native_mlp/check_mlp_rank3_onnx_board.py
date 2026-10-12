#!/usr/bin/env python3
"""Board-only original MLP ONNX versus its actual original Torch capture."""
import argparse, hashlib, json, os, platform, time
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--probe-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
 os.sched_setaffinity(0,{4,5,6,7})
 e=Path('/mnt/vggt-data/vggt-slam2/experiments');capture=e/'vggt-v68-native-mlp-r1'
 assert sha(capture/'result.json')==a.probe_sha256
 r=json.loads((capture/'result.json').read_text());assert r['reconstructed_block_vs_original_full_forward']['bitwise_equal']
 assert r['status']=='completed_component_diagnostic_not_quality_accepted' and r['cleanup_errors']==[]
 a.output.mkdir(parents=True,exist_ok=False)
 report=dict(status='running',script_sha256=sha(__file__),native_capture_sha256=a.probe_sha256,original_model_only=True,
   actual_ORT_on_board_only=True,SDK_executed=False,full_model_or_realtime_validated=False)
 try:
  import numpy as np
  import onnxruntime as ort
  graph=e/'vggt-v68-compiled-mlp-r2/original-frame0-mlp.onnx';assert sha(graph)=='710010d4d687ab27e04025b259e8c1a52601b975fe9e3be80df006f0edf52bf6'
  arrays={}
  for key in ('input','torch_output'):
   row=r['artifacts'][key];path=capture/row['file'];assert sha(path)==row['sha256'];arrays[key]=np.load(path,allow_pickle=False)
   assert arrays[key].shape==(106,1024) and arrays[key].dtype==np.float32
  options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
  options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
  started=time.monotonic();y,=session.run(['output'],{'input':arrays['input'].reshape(2,53,1024)});elapsed=time.monotonic()-started
  assert y.shape==(2,53,1024);y=y.reshape(106,1024)
  assert y.shape==(106,1024) and y.dtype==np.float32 and np.isfinite(y).all()
  delta=y.astype(np.float64)-arrays['torch_output'].astype(np.float64);ref=arrays['torch_output'].astype(np.float64)
  outside=int(np.count_nonzero(np.abs(delta)>1e-5+1e-4*np.abs(ref)))
  np.save(a.output/'output.npy',y,allow_pickle=False)
  report.update(status='checked_original_MLP_FP32_ONNX',onnx_sha256=sha(graph),ORT_version=ort.__version__,
    strict_pass=outside==0,outside=outside,elements=y.size,max_abs=float(np.max(np.abs(delta))),
    relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(ref)),rtol=1e-4,atol=1e-5,
    output_sha256=sha(a.output/'output.npy'),CPU_ORT_seconds_not_NPU_timing=elapsed)
  assert outside==0,'Original FP32 ONNX accuracy gate failed'
 except BaseException as ex:report.update(status='failed',error=f'{type(ex).__name__}: {ex}');raise
 finally:
  (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
