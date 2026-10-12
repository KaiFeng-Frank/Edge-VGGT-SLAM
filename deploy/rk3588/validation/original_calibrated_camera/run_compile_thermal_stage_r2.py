#!/usr/bin/env python3
"""Bound one compiler process; pause its process group for cooling, never change clocks."""
import argparse,json,os,re,signal,subprocess,time
from pathlib import Path
def interrupted(signum, frame):
 raise InterruptedError('Compiler controller received signal ' + str(signum))
for signum in (signal.SIGTERM, signal.SIGHUP):
 signal.signal(signum, interrupted)
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--timeout',type=float,default=180);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
assert a.stage.replace('-','').isalnum() and 0<a.timeout<=900
command=a.command[1:] if a.command and a.command[0]=='--' else a.command
assert command
base=Path('/mnt/vggt-data/vggt-slam2/experiments');record=base/f'vggt-v82-{a.stage}-control-r1.json';log=base/f'vggt-v82-{a.stage}-r1.log'
assert not record.exists() and not log.exists()
def sample():
 return {'elapsed_seconds':time.monotonic()-started,'npu_hz':int(Path('/sys/class/devfreq/fdab0000.npu/cur_freq').read_text()),'temperatures_c':{(p/'type').read_text().strip():int((p/'temp').read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*')}}
def memory_sample(pid):
 result={'pid':pid}
 for name in ('status','smaps_rollup'):
  try:
   fields={}
   for line in (Path('/proc')/str(pid)/name).read_text().splitlines():
    key,sep,value=line.partition(':')
    if sep and (key.startswith(('Vm','Rss','Pss','Private_','Shared_')) or key in ('Anonymous','Swap','SwapPss','Locked')):
     fields[key]=value.strip()
   result[name]=fields
  except (FileNotFoundError,ProcessLookupError,PermissionError) as exc:result[name+'_error']=type(exc).__name__
 try:
  mapping=None;maps=[]
  for line in (Path('/proc')/str(pid)/'smaps').read_text().splitlines():
   if re.match(r'^[0-9a-f]+-[0-9a-f]+ ',line):
    parts=line.split(maxsplit=5);mapping=None
    if len(parts)==6 and ('model.pt' in parts[5] or parts[5].endswith('.rknn')):
     mapping={'header':line};maps.append(mapping)
   elif mapping is not None:
    key,sep,value=line.partition(':')
    if sep and key in ('Size','Rss','Pss','Private_Clean','Private_Dirty','Shared_Clean','Shared_Dirty','Anonymous','Swap'):mapping[key]=value.strip()
  result['weight_mappings']=maps
 except (FileNotFoundError,ProcessLookupError,PermissionError) as exc:result['smaps_error']=type(exc).__name__
 return result
started=time.monotonic();child=None;paused=False;report={'thermal_pause_threshold_c':70,'thermal_resume_threshold_c':63,'compile_scheduler_events':[],'stage':a.stage,'command':command,'status':'running','frequency_written':False,'timeout_seconds':a.timeout,'samples':[],'cleanup_errors':[]}
try:
 initial=sample();report['samples'].append(initial);assert initial['npu_hz']==300000000 and max(initial['temperatures_c'].values())<75
 with log.open('x') as stream:
  child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  report['child_pid']=child.pid
  while child.poll() is None:
   row=sample();row['worker_memory']=memory_sample(child.pid);report['samples'].append(row)
   if row['npu_hz']!=300000000 or max(row['temperatures_c'].values())>=75 or row['elapsed_seconds']>a.timeout:raise RuntimeError('Clock/temperature/timeout observation limit')
   hot=max(row['temperatures_c'].values())
   if not paused and hot>=70:
    os.killpg(child.pid,signal.SIGSTOP);paused=True;report['compile_scheduler_events'].append({'event':'pause_compiler','elapsed_seconds':row['elapsed_seconds'],'temperature_c':hot})
   elif paused and hot<=63:
    os.killpg(child.pid,signal.SIGCONT);paused=False;report['compile_scheduler_events'].append({'event':'resume_compiler','elapsed_seconds':row['elapsed_seconds'],'temperature_c':hot})
   time.sleep(.25)
  report['returncode']=child.returncode;report['status']='completed' if child.returncode==0 else 'failed_child'
except BaseException as exc:
 report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
finally:
 if child is not None and child.poll() is None:
  try:
   os.killpg(child.pid,signal.SIGCONT)
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
  except BaseException as exc:report['cleanup_errors'].append(str(exc))
 if child is not None:report['returncode']=child.poll()
 try:report['samples'].append(sample())
 except BaseException as exc:report['cleanup_errors'].append('Final observation: '+str(exc))
 report['seconds']=time.monotonic()-started
 record.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report.get(k) for k in ('stage','status','returncode','seconds','error','cleanup_errors')}),flush=True)
raise SystemExit(0 if report['status']=='completed' and not report['cleanup_errors'] else 1)
