#!/usr/bin/env python3
"""One real B2D6 saved-state SDK ABBA diagnostic; no feedback/BA/Torch/ORT.

Frozen factory validates model ancestry. A uses original FP32 SDK packing;
B queries its own context and packs owned native FP16. Original output handling
is retained. Public code, model and weight license scopes remain separate.
"""
import argparse,hashlib,importlib.util,json,os,platform,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
FACTORY='c829269de17d775ef2df7663c561272025d3a080c671c01de92bd99ecf3939cf'
CONFIG='0c9005f92cbfda224028624bb0ccdedb705bdb520231021baeedce67cf10dc9c'
BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
KEY='B2_D6_M16'
def require(v,m):
 if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(v):return dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes,tensor_sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest())
def platform_sample():
 value=dict(monotonic=time.monotonic(),thermal_mC={},cpu_kHz={})
 for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
  try:value['thermal_mC'][p.parent.name]=int(p.read_text())
  except (OSError,ValueError):pass
 for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'):
  try:value['cpu_kHz'][p.parent.name]=int(p.read_text())
  except (OSError,ValueError):pass
 return value
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('factory','config','transport','packer','packing-library','query','feed','feed-receipt','build-receipt','conversion-receipt'):
  p.add_argument('--'+n,type=Path,required=True);p.add_argument('--'+n+'-sha256',required=True)
 p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--diagnostic-allow-native-fp16',action='store_true');p.add_argument('--diagnostic-allow-fp32-rejected',action='store_true')
 a=p.parse_args()
 require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board only')
 require(a.diagnostic_allow_native_fp16 and a.diagnostic_allow_fp32_rejected,'Both explicit research gates required')
 out=a.output_dir.resolve();require(out.is_relative_to(BASE),'Isolated board experiments output required');out.mkdir(exist_ok=False)
 report=dict(schema='v43_native_single_B2D6_saved_state_ABBA',status='failed',candidate_accepted=False,original_VGGT_default_unchanged=True,original_strict_FP32_rejected=True,strict_FP32_thresholds_relaxed=False,recurrent_state_committed=False,Torch_ORT_BA_executed=False,all_model_execution_on_RK3588S=True,physical_camera=False,zero_copy_claimed=False,model_key=KEY,calls=[],cleanup_errors=[])
 pins={};sessions={};owner=None;affinity=os.sched_getaffinity(0)
 def pin(path,expected):
  path=Path(path).resolve(strict=True);require(sha(path)==expected,'SHA differs: '+path.name);pins[str(path)]=expected;return path
 def load(path,expected,label):
  path=pin(path,expected);name='_v43_single_'+label+'_'+uuid.uuid4().hex;spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
 def save(name,values):
  f=out/name;np.savez_compressed(f,**values);return dict(file=name,sha256=sha(f),bytes=f.stat().st_size,arrays={k:meta(v) for k,v in values.items()})
 try:
  pin(__file__,sha(__file__))
  for n in ('factory','config','transport','packer','packing_library','query','feed','feed_receipt','build_receipt','conversion_receipt'):pin(getattr(a,n),getattr(a,n+'_sha256'))
  require(a.factory_sha256==FACTORY and a.config_sha256==CONFIG,'Frozen nine-model factory/config required')
  os.sched_setaffinity(0,{4,5,6,7});report['affinity_before']=sorted(affinity);report['affinity_during']=sorted(os.sched_getaffinity(0))
  report['health_start']=platform_sample()
  config=json.loads(a.config.read_text());source=json.loads(a.feed_receipt.read_text());report['feed_selection']=source
  with np.load(a.feed,allow_pickle=False) as z:feed={k:np.array(z[k],copy=True,order='C') for k in z.files}
  require(set(feed)=={'net','inp','corr'},'Three original group inputs required')
  for n,c in [('net',384),('inp',384),('corr',882)]:
   require(feed[n].shape==(2,c,6,16) and feed[n].dtype==np.float32 and np.isfinite(feed[n]).all(),'Finite B2D6 CF input required');feed[n].flags.writeable=False
  before={k:meta(v) for k,v in feed.items()};report['input_arrays']=before;report['feed_sha256']=sha(a.feed)
  require(source['status']=='saved_array_integrity_verified_no_neural_execution' and source['output']['sha256']==a.feed_sha256 and source['extraction']['complete_source_edges_verified'] is True,'Saved original source group provenance differs')
  for n,m in before.items():require(all(source['output']['arrays'][n][k]==v for k,v in m.items()),'Feed metadata differs: '+n)
  conversion=json.loads(a.conversion_receipt.read_text());require(conversion['status']=='passed_conversion_reference_and_ownership_only' and conversion['library_sha256']==a.packing_library_sha256 and conversion['saved_feed']['source_npz_sha256']==a.feed_sha256,'Conversion prerequisite differs')
  report['conversion_receipt_sha256']=a.conversion_receipt_sha256
  build=json.loads(a.build_receipt.read_text());require(build['status']=='compiled_ABI_loaded_conversion_not_yet_verified' and build['output']['sha256']==a.packing_library_sha256 and build['sources_unchanged'] is True,'Packing build provenance differs')
  factory=load(a.factory,FACTORY,'factory');folder=out/'A-native';folder.mkdir()
  def forbidden(*args,**kwargs):raise RuntimeError('No motion probe or CPU fallback allowed')
  owner=factory.open_board_online_from_index(**config,failure_dir=folder,motion_probe=forbidden,diagnostic_allow_fp32_rejected=True)
  pins.update(owner.source_pins);spec=owner.cache.specs[KEY];report['validated_model_spec']=spec
  transport=load(a.transport,a.transport_sha256,'transport')
  runtime=transport.load_runtime(config['parent_runtime'],config['contracts'],config['contracts_sha256'],config['parent_contracts'],runtime_bridge=config['runtime_bridge'],packer_path=a.packer,packer_sha256=a.packer_sha256,packing_library_path=a.packing_library,packing_library_sha256=a.packing_library_sha256,packing_source_pins={str(a.build_receipt):a.build_receipt_sha256},native_query_receipt=a.query,native_query_sha256=a.query_sha256,diagnostic_allow_native_fp16=True)
  pins.update(runtime.source_pins)
  sessions['A']=owner.cache.factory(spec)
  folder=out/'B-native';folder.mkdir()
  sessions['B']=runtime.GroupedSession.open_board(spec['model_path'],spec['model_sha256'],spec['inputs'],spec['outputs'],**{k:config[k] for k in ('abi_path','header_path','library_path','log_helper_path')},export_strict_FP32_gate_passed=spec['export_strict_FP32_gate_passed'],diagnostic_allow_fp32_rejected=True,diagnostic_allow_native_fp16=True,pins=owner.source_pins,collect_perf_detail=False,log_directory=folder)
  report['initial_contexts']={k:v.metrics() for k,v in sessions.items()}
  first=None;seen={'A':0,'B':0}
  for i,method in enumerate(('A','B','B','A')*2):
   health=platform_sample();start=time.monotonic()
   try:values,record=sessions[method].execute(feed)
   except Exception as error:
    if hasattr(error,'record'):report['failed_group_record']=error.record
    if getattr(error,'raw_outputs',None):report['failed_raw_output']=save('failed-raw-output.npz',error.raw_outputs)
    raise
   wall=time.monotonic()-start
   artifact=save(f'call-{i:02d}-{method}.npz',values)
   raw={k:v.tobytes(order='C') for k,v in values.items()}
   if first is None:first=raw
   equal=first==raw
   row=dict(order=i,method=method,state_method_repeat_index=seen[method],excluded_first_state_method_call=seen[method]==0,actual_execute_wall_seconds=wall,output_artifact=artifact,bitwise_equal_first_A=equal,record=record,health_before=health)
   report['calls'].append(row);seen[method]+=1
   require(equal,'Native or repeat output differs bytewise from same-state original SDK')
   require(record['execution_status']=='completed' and record['outputs_finite'] and record['weight_domain_valid'] and not record['all_outputs_zero'] and record['outputs_released'],'Output/release gate failed')
   require(before=={k:meta(v) for k,v in feed.items()},'Original feed changed')
  fields=('pack_seconds','input_set_seconds','run_api_seconds','outputs_get_seconds','owned_copy_seconds','outputs_release_seconds','forward_seconds')
  report['warm_statistics']={}
  for method in ('A','B'):
   rows=[v for v in report['calls'] if v['method']==method and not v['excluded_first_state_method_call']]
   report['warm_statistics'][method]=dict(n=len(rows),actual_execute_wall_mean_seconds=float(np.mean([v['actual_execute_wall_seconds'] for v in rows])),**{f:float(np.mean([v['record']['timing'][f] for v in rows])) for f in fields})
  report.update(status='completed_single_B2D6_AB_bitwise_equal_diagnostic_not_adopted',A_B_all_outputs_bitwise_equal=True,SDK_native_transport_equivalence_for_this_feed_only=True)
 except Exception as error:
  report.update(error=repr(error),traceback=traceback.format_exc())
  if hasattr(error,'record'):report['exception_record']=error.record
 finally:
  report['closed_contexts']={}
  for k,s in sessions.items():
   try:
    s.close();m=s.metrics();report['closed_contexts'][k]=m;require(m['closed'] and m['context_destroyed'] and not m['cleanup_errors'] and m['current_DMA_bytes']==m['current_model_host_bytes']==0,'Context close failed')
   except Exception as error:report['cleanup_errors'].append(k+': '+repr(error))
  if owner is not None:
   try:owner.close();report['factory_owner_after_close']=owner.metrics()
   except Exception as error:report['cleanup_errors'].append('factory: '+repr(error))
  try:
   after={k:sha(k) for k in pins};report.update(source_pins=pins,source_hashes_after=after,sources_unchanged=after==pins);require(after==pins,'Pinned source changed')
   os.sched_setaffinity(0,affinity);report['affinity_restored']=os.sched_getaffinity(0)==affinity;report['health_end']=platform_sample()
  except Exception as error:report['cleanup_errors'].append(repr(error))
  if report['cleanup_errors']:report['status']='failed'
  (out/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report.get(k) for k in ('status','error','cleanup_errors','warm_statistics')},indent=2))
 return 0 if report['status'].startswith('completed_') else 1
if __name__=='__main__':raise SystemExit(main())
