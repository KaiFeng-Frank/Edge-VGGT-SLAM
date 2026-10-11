from pathlib import Path
import json,hashlib
root=Path('/mnt/vggt-data/vggt-slam2');p=root/'datasets/rgbd_dataset_freiburg1_xyz/rgb';files=sorted(p.glob('*.png'))
assert len(files)==798
samples=[]
for split,starts in [('calibration',[32,132,232,332,432,532,632,732]),('evaluation',[80,480])]:
 for k,start in enumerate(starts):
  images=[]
  for index in [start,start+8]:
   f=files[index];images.append({'index':index,'name':f.name,'path':str(f),'timestamp':float(f.stem),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
  samples.append({'sample_id':f'{split}_{k:02d}','split':split,'images':images})
legacy=[]
for index in [0,8]:
 f=files[index];legacy.append({'index':index,'name':f.name,'timestamp':float(f.stem),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
d={'schema':'v49_original_VGGT_real_image_split_plan','dataset':'rgbd_dataset_freiburg1_xyz','dataset_image_count':len(files),'selection':'fixed indices across sequence before inference; no selection by prediction or quality','requested_image_size':336,'expected_tensor_shape':[2,3,252,336],'samples':samples,'legacy_evaluation_images':legacy,'heldout_scope':'Disjoint image files/timestamps within one short TUM sequence, not independent scene or trajectory validation','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out=root/'experiments/vggt-v49-image-plan.json';assert not out.exists();out.write_text(json.dumps(d,indent=2)+'\n');print(hashlib.sha256(out.read_bytes()).hexdigest())
