"""Board-only same-object fingerprint benchmark; original checkpoint, no inference."""
import argparse,hashlib,importlib.util,json,os,resource,statistics,time
from pathlib import Path
B=Path('/mnt/vggt-data/vggt-slam2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--library-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
 assert not a.output.exists();a.output.mkdir(parents=True)
 os.sched_setaffinity(0,{4,5,6,7})
 source=B/'src/vggt_slam/patch_cache.py';source_sha=sha(source);assert source_sha=='3a8cc8c5a68c3bb934969e86053538f0d2026da1846274f4e6085297474a1e4b'
 assert sha(a.library)==a.library_sha256
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from vggt_slam.runtime import load_model
 from vggt_slam.patch_cache import install_patch_cache,PatchCacheHandle
 shim_path=Path(__file__).with_name('native_tensor_states_shim.py')
 assert sha(shim_path)=='cb95a7b521824a28414f0a1c105de4567f7e9e6135a11e305d6eadbfc54e700d'
 spec=importlib.util.spec_from_file_location('_v76_shim',shim_path);shim=importlib.util.module_from_spec(spec);spec.loader.exec_module(shim)
 report=dict(schema='v76_original_model_same_object_native_tensor_fingerprint_benchmark',status='running',original_model_only=True,model_inference_executed=False,SDK_executed=False,checkpoint_identity_verified_by_original_loader=True,checkpoint_sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',source_sha256=source_sha,script_sha256=sha(__file__),shim_sha256=sha(shim_path),library_sha256=a.library_sha256,cpu_affinity=sorted(os.sched_getaffinity(0)),model_threads=1,calls=[],cleanup_errors=[],candidate_accepted=False,real_time_validated=False)
 cache=native=None;started=time.perf_counter();class_method=PatchCacheHandle._fingerprint
 try:
  model=load_model(B/'weights/model.pt',device='cpu',image_size=112,prune_unused_heads=True,offline=True)
  cache=install_patch_cache(model);original=cache._fingerprint
  expected=original();assert expected==cache._state
  report['state_counts']=[len(v) for v in expected]
  native=shim.install_native_tensor_state_loop(cache,a.library,a.library_sha256);candidate=cache._fingerprint
  assert candidate()==expected and PatchCacheHandle._fingerprint is class_method
  report['load_and_setup_seconds']=time.perf_counter()-started
  with torch.inference_mode():
   for i,route in enumerate('ABBA'*12):
    fn=original if route=='A' else candidate
    tick=time.perf_counter();value=fn();seconds=time.perf_counter()-tick
    assert value==expected and value==cache._state and PatchCacheHandle._fingerprint is class_method
    report['calls'].append(dict(index=i,route=route,seconds=seconds,complete_fingerprint_equal=True))
  report['summary']={}
  for route in 'AB':
   vals=[v['seconds'] for v in report['calls'] if v['route']==route]
   report['summary'][route]=dict(warm_count=len(vals)-1,mean_seconds=statistics.mean(vals[1:]),min_seconds=min(vals[1:]),max_seconds=max(vals[1:]),first_seconds=vals[0])
  report['status']='completed_fingerprint_component_only'
 except BaseException as exc:
  report.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
 finally:
  for handle in (native,cache):
   if handle is not None:
    try:handle.close()
    except BaseException as exc:report['cleanup_errors'].append(type(exc).__name__+': '+str(exc))
  report.update(native_handle=native.metrics() if native else None,cache=cache.metrics() if cache else None,original_cache_source_unchanged=sha(source)==source_sha,original_class_method_unchanged=PatchCacheHandle._fingerprint is class_method,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seconds=time.perf_counter()-started)
  if report['cleanup_errors']:report['status']='failed'
  (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report.get(k) for k in ['status','summary','cleanup_errors']}),flush=True)
if __name__=='__main__':main()
