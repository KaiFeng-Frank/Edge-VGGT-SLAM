"""Saved NumPy arrays only: no image decode, model, ORT, native execution or SDK."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact-root',type=Path,required=True)
R=parser.parse_args().artifact_root.resolve()
checks=0;files=set()
def ck(v,message):
 global checks
 checks+=1
 if not v:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def meta(a):return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def artifact(folder,row):
 p=folder/row['file'];ck(p.name==row['file'],'basename');ck(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'file identity');files.add(str(p))
 with np.load(p,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
 ck({k:meta(x) for k,x in v.items()}==row['arrays'],'raw metadata');return v
def gate(a,b,row):
 ck(a.shape==b.shape and a.dtype==b.dtype==np.float32,'comparison contract');ck(np.isfinite(a).all() and np.isfinite(b).all(),'finite')
 diff=np.abs(a.astype(np.float64)-b.astype(np.float64));bad=int(np.count_nonzero(diff>1e-5+1e-4*np.abs(b.astype(np.float64))))
 ck(row['rtol']==1e-4 and row['atol']==1e-5 and row['failed_elements']==bad and row['gate_passed']==(bad==0),'strict gate')
 ck(row['max_abs']==float(diff.max()),'max abs')
replaydir=R/'output/vggt-v39-half-torch-capture-r1';replay=json.loads((replaydir/'run.json').read_text())
for key in ('snapshot_before_termination','final_snapshot'):artifact(replaydir,replay[key]['artifact'])
for row in replay['update_captures']:
 x=artifact(replaydir,row['input']);y=artifact(replaydir,row['gold']);ii,jj,kk=(x[k] for k in ('ii','jj','kk'))
 ck(len(ii)==row['edges'] and np.array_equal(ii,kk//16),'full source ownership')
 ck(x['net'].shape==x['inp'].shape==y['net'].shape==(1,len(ii),384) and x['corr'].shape==(1,len(ii),882),'full neural shapes')
 for src in np.unique(ii):
  mask=ii==src
  for dst in np.unique(jj[mask]):ck(np.array_equal(np.sort(kk[mask & (jj==dst)]%16),np.arange(16)),'complete16 patch group')
 ck(all(np.isfinite(v).all() for v in y.values()) and ((y['weight']>=0)&(y['weight']<=1)).all(),'finite valid gold')
probe_dir=R/'output/vggt-v39-half-encoder-probe-r1';probe=json.loads((probe_dir/'probe.json').read_text())
for index,row in enumerate(probe['samples']):
 x=artifact(probe_dir,row['input_artifact']);teacher=artifact(probe_dir,row['teacher_artifact'])
 ck(meta(x['normalized_bgr'])==replay['frames'][index]['normalized_half_input'],'replay/probe exact normalized inputs')
 ck(row['resized_BGR']==replay['frames'][index]['resized_BGR'],'same half pixels')
 yx=x['centers_xy'];ck(np.array_equal(x['patch_indices'],yx[:,1]*76+yx[:,0]),'integer ordered indices')
 context=teacher['imap'][0,:,yx[:,1],yx[:,0]][None]
 ck(np.array_equal(context,teacher['context']),'dense original gather context')
 for method in ('original','reciprocal'):
  actual=artifact(probe_dir,row[method]['artifact'])
  for key,v in actual.items():gate(v,teacher[key],row[method]['versus_new_size_original_CPU'][key])
caldir=R/'output/vggt-v39-half-calibration-r1';cal=json.loads((caldir/'calibration.json').read_text());sampling=artifact(caldir,cal['sampling_artifact'])
ck(cal['calibration_source_indices']==[256+16*i for i in range(32)],'fixed32 source schedule')
ck(not set(cal['calibration_source_indices'])&set(range(192)),'disjoint source indices')
lines=[]
for i,row in enumerate(cal['samples']):
 names=[]
 for item,key,shape,dtype in zip(row['input_files'],('image_array','index_array'),((1,3,232,304),(16,)),(np.float32,np.int64)):
  p=caldir/item['file'];ck(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'calibration file');files.add(str(p));v=np.load(p,allow_pickle=False)
  ck(v.shape==shape and v.dtype==dtype and meta(v)==row[key],'calibration tensor')
  if key=='index_array':ck(np.array_equal(v,sampling['centers_xy'][i,:,1]*76+sampling['centers_xy'][i,:,0]),'sampledcal indices')
  names.append(item['file'])
 lines.append(' '.join(names))
ck((caldir/'dataset.txt').read_text().splitlines()==lines,'ordered dataset list')
ck(np.array_equal(sampling['state_before'][1:],sampling['state_after_depth'][:-1]),'RNG seams')
report=dict(schema='v39_saved_array_audit',status='passed',checks=checks,files_read=len(files),source_sha256=sha(Path(__file__)),receipts={str(p.relative_to(R)):sha(p) for p in (replaydir/'run.json',probe_dir/'probe.json',caldir/'calibration.json')},real_model_ORT_SDK_or_image_decode=False,candidate_accepted=False,realtime_validated=False)
(R/'output/saved-array-audit-r1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
