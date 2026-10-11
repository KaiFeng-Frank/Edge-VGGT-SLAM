#!/usr/bin/env python3
"""Board-only original 24-block internal DINO PTQ diagnostic; graphs and activations stay private."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--capture-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',type=int);p.add_argument('--count',type=int,default=24);a=p.parse_args()
assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes();os.sched_setaffinity(0,{4,5,6,7})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(4<<20),b''):h.update(z)
 return h.hexdigest()
def artifact(root,row):
 p=(root/row['file']).resolve(strict=True);assert p.is_relative_to(root.resolve()) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];return p
assert sha(a.manifest)==a.manifest_sha256 and sha(a.capture)==a.capture_sha256
m=json.loads(a.manifest.read_text());c=json.loads(a.capture.read_text())
assert m['schema']=='v54_original_VGGT_24_DINO_block_RKNN_manifest' and c['status']=='completed_original_FP32_calibration_capture' and not c['cleanup_errors']
assert c['checkpoint']==m['checkpoint'] and c['full_forward_completed']==8 and len(m['blocks'])==24 and 1<=a.count<=24
assert [g['index'] for g in c['groups']['dino']]==list(range(24))
if a.worker is not None:
 i=a.worker;row=m['blocks'][i];g=c['groups']['dino'][i];assert row['role']=='dino' and row['index']==i and len(g['samples'])==16
 graph=artifact(a.manifest.parent,row);dataset=artifact(a.capture.parent,g['dataset']);items=[artifact(a.capture.parent,r['artifact']) for r in g['samples']]
 assert dataset.read_text().splitlines()==[str(p) for p in items]
 folder=a.output/f'dino-{i:02d}';folder.mkdir(parents=True,exist_ok=False);os.chdir(folder);os.environ['TMPDIR']=str(folder)
 from rknn.api import RKNN
 config=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,mean_values=None,std_values=None,quant_img_RGB2BGR=False,model_pruning=False,remove_reshape=False,quantize_weight=False,enable_flash_attention=False,quantized_dtype='w8a8',quantized_algorithm='normal',quantized_method='channel')
 report=dict(index=i,original_graph_sha256=row['sha256'],capture_sha256=a.capture_sha256,dataset_sha256=g['dataset']['sha256'],configuration=config,operations=[],status='running',candidate_accepted=False)
 sdk=RKNN(verbose=False,verbose_file=str(folder/'sdk.log'));start=time.monotonic()
 try:
  for name,fn in [('config',lambda:sdk.config(**config)),('load',lambda:sdk.load_onnx(model=str(graph))),('build',lambda:sdk.build(do_quantization=True,dataset=str(dataset),auto_hybrid=False)),('export',lambda:sdk.export_rknn(str(folder/'block.rknn')))]:
   t=time.monotonic();code=fn();report['operations'].append(dict(name=name,code=code,seconds=time.monotonic()-t));assert code in (0,None),(name,code)
  p=folder/'block.rknn';report.update(status='compiled_not_quality_accepted',rknn=dict(file=str(p.relative_to(a.output)),bytes=p.stat().st_size,sha256=sha(p)))
 except BaseException as e:report.update(status='failed',error=str(e));raise
 finally:sdk.release();report['seconds']=time.monotonic()-start;(folder/'compile.json').write_text(json.dumps(report,indent=2)+'\n')
else:
 a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
 for i in range(a.count):
  subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--manifest',str(a.manifest),'--manifest-sha256',a.manifest_sha256,'--capture',str(a.capture),'--capture-sha256',a.capture_sha256,'--output',str(a.output),'--count',str(a.count),'--worker',str(i)],check=True)
  r=json.loads((a.output/f'dino-{i:02d}/compile.json').read_text());assert r['status']=='compiled_not_quality_accepted';m['blocks'][i]['rknn']=r['rknn'];print(json.dumps(dict(completed=i+1,total=a.count,seconds=time.monotonic()-begin)),flush=True)
 m.update(schema='v57_original_VGGT_calibrated_DINO_RKNN_manifest',compiled_count=a.count,parent_manifest_sha256=a.manifest_sha256,parent_manifest_directory=str(a.manifest.parent),capture_sha256=a.capture_sha256,calibration_directory=str(a.capture.parent),status='compiled_not_quality_accepted',candidate_accepted=False,seconds=time.monotonic()-begin)
 (a.output/'compiled.json').write_text(json.dumps(m,indent=2)+'\n')
 print(json.dumps(dict(status=m['status'],sha256=sha(a.output/'compiled.json'))),flush=True)
