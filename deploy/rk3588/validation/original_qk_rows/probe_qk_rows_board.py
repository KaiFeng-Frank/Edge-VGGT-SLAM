#!/usr/bin/env python3
"""Original W8A8 pair00 versus a layout-only frame Q/K rewrite on RK3588.

Use the existing original calibrated SDK route with FP32 conversion, owned
copies, release and finite checks. Profile runs remain separate from timing.
"""
from __future__ import annotations
import argparse, ctypes as ct, hashlib, importlib.util, json, math, os
from pathlib import Path
import re, sys, time
from types import SimpleNamespace
BASE=Path('/mnt/vggt-data/vggt-slam2'); E=BASE/'experiments'
ABI_SHA='37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HELPER_SHA='f3801ba5b92c1043aab2d3ebc6c0ad3258b8633c25e216b1298fbb1e0c9c38a0'
IO_SHA='f3f106034c6621a1e3ed07bdb265107b8522103f7d186ba4bbe0d0cefda16f2f'
BASELINE_SHA='bdfe3757a7b4adef891dd622525d8a70ba948e9c0af9d0c1d9fffed41a74ea91'
BASELINE_LEAF_SHA='46e83eb3d2b0e3ddfe7176d0d7de456190b53c48a9315a50bf0ea551315c9f8f'
GRAPH_SHA='efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
CAPTURE_SHA='a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
CONFIG=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,
 mean_values=None,std_values=None,quant_img_RGB2BGR=False,model_pruning=False,
 remove_reshape=False,quantize_weight=False,enable_flash_attention=False,
 quantized_dtype='w8a8',quantized_algorithm='normal',quantized_method='channel')
PUBLIC=[dict(index=0,name='frame_output',shape=[2,53,1024],dtype='float32'),
 dict(index=1,name='global_output',shape=[1,106,1024],dtype='float32')]
def require(ok,message):
 if not ok: raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=Path,default=BASE/'src')
 p.add_argument('--compiled',type=Path,required=True);p.add_argument('--compiled-sha256',required=True)
 p.add_argument('--compiler-source',type=Path,required=True);p.add_argument('--compiler-sha256',required=True)
 p.add_argument('--check',type=Path,required=True);p.add_argument('--check-sha256',required=True)
 p.add_argument('--check-source',type=Path,required=True);p.add_argument('--check-source-sha256',required=True)
 p.add_argument('--io-source',type=Path,default=E/'vggt-v80-source-r1/rknn_calibrated_retained.py')
 p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--profile',action='store_true')
 a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 pins,contexts,owners={},[],[];api=base=None
 affinity=os.sched_getaffinity(0);old_path=sys.path[:]
 envkeys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
 old_env={k:os.environ.get(k) for k in envkeys}
 r=dict(schema='v86_original_W8A8_pair00_QK_rows_component',status='failed',script_sha256=sha(__file__),
  original_model_only=True,Torch_model_or_checkpoint_constructed_or_loaded=False,host_GPU_used=False,
  profile=a.profile,normal_benchmark_timing=not a.profile,init_flags=8 if a.profile else 0,core_mask=7,
  schedule='AB' if a.profile else 'AB'+'ABBA'*3,hidden_warmup_calls=0,
  input_type='FP32 SDK-converted; unchanged original calibrated route',output_want_float=1,
  API_conversion_not_internal_FP32=True,all_internal_ops_INT8_verified=False,
  clock_writes=False,expected_npu_hz=1000000000,candidate_accepted=False,quality_accepted=False,
  complete_model_or_trajectory_or_realtime_validated=False,calls=[],queries=[],cleanup_errors=[],
  contexts_created=0,contexts_destroyed=0,strict_rtol=1e-4,strict_atol=1e-5,
  timing_scope='Original finite input check, SDK input conversion/run/output conversion, owned FP32 copy, release and finite outputs; excludes hashes/saves/comparisons/profile queries.')
 def pin(path,digest):
  path=Path(path).resolve(strict=True);require(sha(path)==digest,'Frozen file differs: '+str(path));pins[path]=digest;return path
 def artifact(root,row):
  root=Path(root).resolve(strict=True);rel=Path(row['file']);require(not rel.is_absolute() and '..' not in rel.parts,'Relative artifact required')
  path=(root/rel).resolve(strict=True);require(path.is_relative_to(root) and path.stat().st_size==row['bytes'],'Artifact path/size differs');return pin(path,row['sha256'])
 def load(path,digest,name):
  spec=importlib.util.spec_from_file_location(name,pin(path,digest));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 try:
  pin(__file__,r['script_sha256'])
  validation=a.repo/'deploy/rk3588/validation';base=load(validation/'run-v12-rknn-block-board.py',ABI_SHA,'v86_ABI');r['board']=base.require_board()
  require({4,5,6,7}<=affinity,'CPUs4..7 required');os.sched_setaffinity(0,{4,5,6,7});os.environ.update({k:'1' for k in envkeys});sys.path.insert(0,str(a.repo))
  import numpy as np
  helper=load(validation/'original_pair_group/probe_original_pair_group.py',HELPER_SHA,'v86_pair_IO')
  io=load(a.io_source,IO_SHA,'v86_original_calibrated_IO')
  pin(a.repo/'vggt_slam/rknn_retained_aggregator.py','04fbb5b2ad7db22bf26d52d4a6b5a3fadffa042557919575938b035f5dd06b5f')
  pin(a.repo/'vggt_slam/rknn_aggregator_blocks.py','4fea665334e9d12a3d55839c3150680e269bab9794074d698bed2ad0a6e41c93')
  require(Path(io.parent.__file__).resolve()==(a.repo/'vggt_slam/rknn_retained_aggregator.py').resolve(),'IO parent escaped source root')
  pin(a.compiler_source,a.compiler_sha256);pin(a.check_source,a.check_source_sha256)
  compiled=json.loads(pin(a.compiled,a.compiled_sha256).read_text());gate=json.loads(pin(a.check,a.check_sha256).read_text())
  require(compiled['schema']=='v86_original_pair00_QK_rows_W8A8_compile'
   and compiled['status']=='compiled_not_quality_accepted' and compiled['script_sha256']==a.compiler_sha256
   and compiled['configuration']==CONFIG and compiled['released'] is True and compiled['cleanup_errors']==[]
   and all(compiled[k] for k in ('source_and_inputs_unchanged','environment_restored','affinity_restored')),
   'Successful unchanged original pair layout W8A8 compile required')
  require(gate['schema']=='v86_original_VGGT_pair00_frame_QK_rows_FP32_check'
   and gate['status']=='passed_original_FP32_ONNX_equivalence_not_quality_accepted'
   and gate['script_sha256']==a.check_source_sha256 and gate['original_pair_onnx_sha256']==GRAPH_SHA
   and gate['strict_rtol']==1e-4 and gate['strict_atol']==1e-5
   and gate['all_candidate_vs_original_FP32_ONNX_strict_passed'] is True and gate['cleanup_errors']==[],
   'Original fixed FP32 complete-consumer gate required')
  require(compiled['check_receipt_sha256']==a.check_sha256 and compiled['graph']['sha256']==gate['candidate_onnx_sha256'],
   'Compiler and FP32 check graph identities differ')
  original_root=E/'vggt-v56-ptq-pairs-r1'
  original=json.loads(pin(original_root/'compiled.json',BASELINE_SHA).read_text())
  leaf=json.loads(pin(original_root/'pair-00/compile.json',BASELINE_LEAF_SHA).read_text());block=original['blocks'][0]
  require(original['status']=='compiled_not_quality_accepted' and block['index']==0 and block['role']=='pair'
   and block['sha256']==leaf['original_graph_sha256']==GRAPH_SHA and leaf['configuration']==CONFIG
   and leaf['capture_sha256']==original['capture_sha256']==CAPTURE_SHA
   and block['output_bindings']==PUBLIC and leaf['rknn']==block['rknn']
   and [x['name'] for x in leaf['operations']]==['config','load','build','export']
   and all(x['code']==0 for x in leaf['operations']), 'Matching original calibrated pair00 baseline required')
  calibration_path=pin(E/'vggt-v56-calibration-r1/capture.json',CAPTURE_SHA)
  calibration=json.loads(calibration_path.read_text());dataset=artifact(calibration_path.parent,calibration['groups']['aggregator'][0]['dataset'])
  require(sha(dataset)==leaf['dataset_sha256'],'Frozen original calibration dataset differs')
  require(dataset.read_text()==(a.compiled.parent/'dataset.txt').read_text(),'Exactly the same eight calibration input rows required')
  pin(a.compiled.parent/'dataset.txt',leaf['dataset_sha256'])
  for rel,digest in original['source_hashes'].items():pin(a.repo/rel,digest)
  models=[('A',artifact(original_root,block['rknn'])),('B',artifact(a.compiled.parent,compiled['rknn']))]
  cases=[]
  for row in gate['cases']:
   x=np.load(artifact(a.check.parent,row['input']),allow_pickle=False)
   require(x.dtype==np.float32 and x.shape==(2,53,1024) and np.isfinite(x).all()
    and hashlib.sha256(x.tobytes()).hexdigest()==row['input']['raw_sha256'],'Original real frame input differs')
   refs=[np.load(artifact(a.check.parent,item),allow_pickle=False) for item in row['original_outputs']]
   require(len(refs)==2,'Both complete original pair outputs required')
   cases.append((row,x,refs))
  require([x[0]['tag'] for x in cases]==['v53','v56-calib00'],'Two original inputs required')
  r.update(compile_receipt_sha256=a.compiled_sha256,check_receipt_sha256=a.check_sha256,
   baseline_manifest_sha256=BASELINE_SHA,baseline_leaf_sha256=BASELINE_LEAF_SHA,
   original_graph_sha256=GRAPH_SHA,candidate_graph_sha256=gate['candidate_onnx_sha256'],
   baseline_reused_not_recompiled=True,same_calibration_dataset_sha256=leaf['dataset_sha256'],
   compiler_source_sha256=a.compiler_sha256,FP32_checker_source_sha256=a.check_source_sha256,
   original_calibrated_IO_source_sha256=IO_SHA,cpu_affinity=sorted(os.sched_getaffinity(0)),cases=gate['cases'])
  pin(a.repo/'deploy/rk3588/reference/rknn_api_v2.3.2.h',helper.HEADER_SHA)
  api=base.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so',helper.SDK_SHA))))
  for role,path in models:
   context=base.CONTEXT();contexts.append(context);name=ct.create_string_buffer(str(path).encode())
   code=api.rknn_init(ct.byref(context),ct.cast(name,ct.c_void_p),0,r['init_flags'],None)
   r['contexts_created']+=int(bool(context.value));base.check_call(code,'rknn_init '+role);require(context.value,'Empty RKNN context')
   base.check_call(api.rknn_set_core_mask(context,7),'rknn_set_core_mask')
   count=base.query(api,context,0,base.InputOutputNum());require((count.n_input,count.n_output)==(1,2),'Original 1-input2-output ABI required')
   inp=io.calibrated_io_attr(base,base.query(api,context,1,base.TensorAttr(index=0)),[2,53,1024],'input')
   outs=[io.calibrated_io_attr(base,base.query(api,context,2,base.TensorAttr(index=b['index'])),b['shape'],b['name'],b['index']) for b in PUBLIC]
   version=base.query(api,context,5,base.SDKVersion());require(re.match(r'^2\.3\.2(?:\D|$)',bytes(version.api_version).decode()),'Typed SDK2.3.2 required')
   memory=base.query(api,context,6,base.MemSize())
   stats=dict(sdk_run_attempts=0,sdk_runs_succeeded=0,outputs_acquired=0,outputs_released=0,sdk_io_seconds=0.,cleanup_errors=[],
    fp32_io=dict(calls_attempted=0,calls_completed=0,owned_output_copy_seconds=0.,finite_check_seconds=0.,route_seconds=0.,
     **{k+'_seconds':0. for k in ('sdk_api','rknn_inputs_set','rknn_run','rknn_outputs_get','rknn_outputs_release')}))
   owners.append(SimpleNamespace(base=base,api=api,np=np,timeout=30000,stats=stats))
   r['queries'].append(dict(role=role,model_sha256=sha(path),inputs=[inp],outputs=outs,
    SDK_version=bytes(version.api_version).decode(),driver_version=bytes(version.drv_version).decode(),
    memory=dict(weight_bytes=int(memory.total_weight_size),internal_bytes=int(memory.total_internal_size),dma_bytes=int(memory.total_dma_allocated_size))))
  try:
   clock_lines=Path('/sys/kernel/debug/clk/clk_summary').read_text().splitlines()
   r['clock_summary_before_calls']=[line.strip() for line in clock_lines if re.search(r'(scmi_clk_ddr|scmi_clk_npu|clk_npu_dsu0|aclk_npu[012])',line)]
  except OSError as exc:r['clock_summary_read_error']=str(exc)
  for case,x,refs in cases:
   first={}
   for index,role in enumerate(r['schedule']):
    which=role=='B';owner=owners[which];before=helper.sample();require(before['actual_frequency_hz']==1000000000,'NPU1GHz required')
    previous=dict(owner.stats['fp32_io']);y=io.RKNNCalibratedRetained._run_group(owner,contexts[which],x,PUBLIC)
    timing={k:v-previous[k] for k,v in owner.stats['fp32_io'].items()};after=helper.sample();require(after['actual_frequency_hz']==1000000000,'NPU changed')
    cold=role not in first
    if cold:first[role]=y
    path=out/f'{case["tag"]}-{index:02d}-{role}.npz'
    with path.open('xb') as f:np.savez(f,frame_output=y[0],global_output=y[1])
    call=dict(input=case['tag'],sequence=index,role=role,first_for_input_role=cold,component_seconds=timing['route_seconds'],timing=timing,
     platform_before=before,platform_after=after,vs_original_FP32_ONNX=[helper.compare(np,v,ref) for v,ref in zip(y,refs)],
     repeat_first_role=[helper.compare(np,v,ref) for v,ref in zip(y,first[role])],
     vs_first_A=[helper.compare(np,v,ref) for v,ref in zip(y,first['A'])],artifact=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)))
    if a.profile:
     profile=base.perf_detail(api,contexts[which]);name=f'{case["tag"]}-{role}-profile.txt';(out/name).write_text(profile.pop('raw_text'))
     profile.update(file=name,file_sha256=sha(out/name));call['profile']=profile
    r['calls'].append(call)
   require(hashlib.sha256(x.tobytes()).hexdigest()==case['input']['raw_sha256'],'Source input mutated')
  if not a.profile:
   r['summary']={role:dict(warm_count=12,mean_component_seconds=float(np.mean([c['component_seconds'] for c in r['calls'] if c['role']==role and not c['first_for_input_role']])),
    per_input={case['tag']:float(np.mean([c['component_seconds'] for c in r['calls'] if c['role']==role and c['input']==case['tag'] and not c['first_for_input_role']])) for case,_,_ in cases}) for role in 'AB'}
  r['status']='completed_component_diagnostic_not_quality_accepted'
 except BaseException as exc:r['error']=type(exc).__name__+': '+str(exc)
 finally:
  for context in reversed(contexts):
   if context.value:
    try:base.check_call(api.rknn_destroy(context),'rknn_destroy');context.value=0;r['contexts_destroyed']+=1
    except BaseException as exc:r['cleanup_errors'].append(str(exc))
  r['live_contexts']=sum(bool(x.value) for x in contexts);r['route_stats']=[x.stats for x in owners]
  try:r['source_and_inputs_unchanged']=bool(pins) and all(sha(p)==s for p,s in pins.items())
  except BaseException as exc:r['source_and_inputs_unchanged']=False;r['cleanup_errors'].append(str(exc))
  try:
   os.sched_setaffinity(0,affinity);r['affinity_restored']=os.sched_getaffinity(0)==affinity
   for k,v in old_env.items():
    if v is None:os.environ.pop(k,None)
    else:os.environ[k]=v
   sys.path[:]=old_path;r['environment_restored']=all(os.environ.get(k)==v for k,v in old_env.items())
  except BaseException as exc:r['cleanup_errors'].append(str(exc))
  if r['status'].startswith('completed'):
   runs=2 if a.profile else 14
   if not (len(owners)==2 and all(x.stats['sdk_run_attempts']==x.stats['sdk_runs_succeeded']==runs and x.stats['outputs_acquired']==x.stats['outputs_released']==runs*2 and not x.stats['cleanup_errors'] for x in owners)):
    r['cleanup_errors'].append('SDK run/buffer accounting failed')
  if r['cleanup_errors'] or r['live_contexts'] or not all(r.get(k) for k in ('source_and_inputs_unchanged','affinity_restored','environment_restored')):r['status']='failed_cleanup_or_identity'
  (out/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
  print(json.dumps({k:r.get(k) for k in ('status','error','summary','cleanup_errors')}),flush=True)
 return 0 if r['status'].startswith('completed') else 1
if __name__=='__main__':raise SystemExit(main())
