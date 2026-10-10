#!/usr/bin/env python3
"""Reproduce the board source-only JPEG/optical-flow probe without SLAM.

Source deploy/rk3588/env.sh first. Deployment paths and the 30 fps rate match
this recorded probe; an existing output report is deliberately not overwritten.
"""

import hashlib,json,math,re,resource,time
from pathlib import Path
setup_started=time.monotonic()
import cv2,numpy as np,torch
from vggt_slam.cameras import ReplayCamera
from vggt_slam.frame_overlap import FrameTracker
root=Path('/mnt/vggt-data/vggt-slam2')
output=root/'outputs/replay-camera-office473-source-only.json'
if output.exists(): raise FileExistsError(str(output))
paths=sorted((root/'datasets/office_loop').glob('*.jpg'),key=lambda p:int(re.search(r'frame_(\d+)',p.name).group(1)))
if len(paths)!=473: raise RuntimeError('Expected all 473 original JPEGs')
source_sha={}
for relative in ['vggt_slam/cameras.py','vggt_slam/frame_overlap.py']:
 with (root/'src'/relative).open('rb') as handle: source_sha[relative]=hashlib.file_digest(handle,'sha256').hexdigest()
def sensors():
 values={}
 for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
  try: values[(zone/'type').read_text().strip()]=int((zone/'temp').read_text())/1000.0
  except (OSError,ValueError): pass
 return values
def summary(values):
 values=np.asarray(values,dtype=np.float64)
 if not values.size: return None
 return dict(count=int(values.size),min=float(values.min()),mean=float(values.mean()),median=float(np.median(values)),
             p95=float(np.percentile(values,95)),max=float(values.max()),total=float(values.sum()))
cv2.setNumThreads(1)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
camera=ReplayCamera(paths,fps=30,timeout=5)
tracker=FrameTracker()
report={'status':'running','source_type':'paced_image_replay','replay_timing_synthetic':True,
        'pipeline':'ReplayCamera + real OpenCV FrameTracker; source-only',
        'slam_ran':False,'vggt_called':False,'npu_called':False,'slam_validated':False,
        'live_pipeline_validated':False,'real_time_validated':False,
        'input_frames':473,'requested_fps':30,'min_disparity':0,'cv2_threads':1,
        'torch_threads':4,'source_sha256':source_sha,'thermal_at_start_c':sensors(),
        'setup_seconds':time.monotonic()-setup_started}
delivered_ids=[]; accepted_ids=[]; ages=[]; waits=[]; flow_times=[]; shapes=set()
begin=time.monotonic()
try:
 camera.start()
 while True:
  tick=time.monotonic()
  frame=camera.capture()
  delivered=time.monotonic()
  if frame is None:
   if camera.exhausted: break
   continue
  stamp=camera.frame_timestamp
  age=delivered-stamp
  if not math.isfinite(age) or age<0: raise RuntimeError('Invalid software arrival age')
  frame_id=int(re.search(r'frame_(\d+)',Path(camera.frame_path).name).group(1))
  delivered_ids.append(frame_id); ages.append(age); waits.append(delivered-tick); shapes.add(tuple(frame.shape))
  tick=time.monotonic()
  accepted=tracker.compute_disparity(frame,0)
  flow_times.append(time.monotonic()-tick)
  if accepted: accepted_ids.append(frame_id)
 report['eof_observed']=camera.exhausted
 report['status']='passed'
except BaseException as exc:
 report['status']='failed';report['error']=str(type(exc).__name__)+': '+str(exc)
finally:
 end=time.monotonic()
 camera.stop()
 source=camera.acquisition_stats
 report.update(source=source,source_seconds=end-begin,source_frames_delivered=len(delivered_ids),
               accepted_keyframes=len(accepted_ids),first_delivered_frame_id=delivered_ids[0] if delivered_ids else None,
               last_delivered_frame_id=delivered_ids[-1] if delivered_ids else None,
               delivered_frame_ids=delivered_ids,flow_accepted_frame_ids=accepted_ids,
               source_frame_shapes=[list(shape) for shape in sorted(shapes)],
               software_buffer_age_seconds=summary(ages),capture_wait_seconds=summary(waits),
               optical_flow_seconds=summary(flow_times),thermal_at_end_c=sensors(),
               peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 report['source_delivery_fps']=len(delivered_ids)/(end-begin)
 first,last=source.get('first_frame_monotonic'),source.get('last_frame_monotonic')
 report['source_read_fps']=(source['frames_read']-1)/(last-first) if first is not None and last is not None and last>first else None
 report['read_counter_balance_ok']=source['frames_read']==source['frames_delivered']+source['frames_overwritten']
 report['consumer_delivery_counter_ok']=source['frames_delivered']==len(delivered_ids)
 report['unique_chronological_deliveries']=len(set(delivered_ids))==len(delivered_ids) and all(a<b for a,b in zip(delivered_ids,delivered_ids[1:]))
 if report['status']=='passed' and not (report.get('eof_observed') and source['frames_read']==473 and
       report['last_delivered_frame_id']==473 and report['read_counter_balance_ok'] and
       report['consumer_delivery_counter_ok'] and report['unique_chronological_deliveries']):
  report['status']='failed'; report['error']='EOF, final-frame or source counter invariant failed'
 output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(report,allow_nan=False))
if report['status']!='passed': raise RuntimeError(report['error'])
