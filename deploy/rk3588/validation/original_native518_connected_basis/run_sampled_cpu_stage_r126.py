#!/usr/bin/env python3
"""Bound one compiler process; pause its process group for cooling, never change clocks."""
import argparse,json,os,re,resource,signal,subprocess,time
from pathlib import Path

def process_group_members(pgid,proc_root=Path('/proc')):
 """Return live members of the exact new-session process group; zombies excluded."""
 members=[]
 for entry in proc_root.iterdir():
  if not entry.name.isdecimal():continue
  try:
   raw=(entry/'stat').read_text();tail=raw[raw.rfind(')')+2:].split()
   pid=int(raw.split(' ',1)[0]);pgrp=int(tail[2]);session=int(tail[3])
   if pgrp==pgid and session==pgid and tail[0]!='Z':
    members.append({'pid':pid,'ppid':int(tail[1]),'pgrp':pgrp,'session':session,'state':tail[0],'start_time_ticks':int(tail[19])})
  except (FileNotFoundError,ProcessLookupError):continue
 return sorted(members,key=lambda row:row['pid'])

def cleanup_process_group(child,grace_seconds=5.0,kill_wait_seconds=5.0):
 """Stop every live owned-session member, even after the session leader exited."""
 if child.pid<=1 or child.pid in (os.getpgrp(),os.getsid(0)):
  raise ValueError('Refuse to signal the controller or a nonpositive process group')
 began=time.monotonic();result={'pgid':child.pid,'session':child.pid,'grace_seconds':grace_seconds,
  'kill_wait_seconds':kill_wait_seconds,'snapshots':[],'signals':[],'errors':[],'scan_count':0}
 def snapshot(phase):
  members=process_group_members(child.pid);result['scan_count']+=1
  result['snapshots'].append({'phase':phase,'elapsed_seconds':time.monotonic()-began,'members':members})
  return members
 def send(sig):
  members=process_group_members(child.pid);result['scan_count']+=1
  event={'signal':signal.Signals(sig).name,'elapsed_seconds':time.monotonic()-began,'members_before_signal':members}
  if not members:event['status']='not_needed'
  else:
   try:os.killpg(child.pid,sig);event['status']='sent'
   except ProcessLookupError:event['status']='group_disappeared'
   except BaseException as exc:
    event.update(status='failed',error=f'{type(exc).__name__}: {exc}');result['errors'].append(event['error'])
  result['signals'].append(event)
 initial=snapshot('before_cleanup');result['leader_returncode_before_cleanup']=child.poll()
 result['leader_exited_with_live_members']=result['leader_returncode_before_cleanup'] is not None and bool(initial)
 if initial:
  send(signal.SIGCONT);send(signal.SIGTERM);deadline=time.monotonic()+grace_seconds
  while time.monotonic()<deadline:
   child.poll();members=process_group_members(child.pid);result['scan_count']+=1
   if not members:break
   time.sleep(min(.05,max(0,deadline-time.monotonic())))
  remaining=snapshot('after_TERM_grace')
  if remaining:
   send(signal.SIGKILL);deadline=time.monotonic()+kill_wait_seconds
   while time.monotonic()<deadline:
    child.poll();members=process_group_members(child.pid);result['scan_count']+=1
    if not members:break
    time.sleep(min(.05,max(0,deadline-time.monotonic())))
 try:child.wait(timeout=kill_wait_seconds);result['leader_reaped']=True
 except BaseException as exc:
  result['leader_reaped']=False;result['errors'].append('Leader reap: '+f'{type(exc).__name__}: {exc}')
 result['remaining_live_members']=snapshot('after_cleanup');result['live_group_gone']=not result['remaining_live_members']
 result['leader_returncode']=child.poll();result['cleanup_seconds']=time.monotonic()-began
 if not result['live_group_gone']:result['errors'].append('Live original process-group members remain after SIGKILL')
 return result

def interrupted(signum, frame):
 raise InterruptedError('Compiler controller received signal ' + str(signum))
for signum in (signal.SIGTERM, signal.SIGHUP):
 signal.signal(signum, interrupted)
p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--expected-npu-hz',type=int,required=True);p.add_argument('--timeout',type=float,default=180);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
assert a.stage.replace('-','').isalnum() and 0<a.timeout<=900
command=a.command[1:] if a.command and a.command[0]=='--' else a.command
assert command
base=Path('/mnt/vggt-data/vggt-slam2/experiments');record=base/f'vggt-v83-{a.stage}-control-r1.json';log=base/f'vggt-v83-{a.stage}-r1.log'
assert not record.exists() and not log.exists()
def sample():
 return {'elapsed_seconds':time.monotonic()-started,'npu_hz':int(Path('/sys/class/devfreq/fdab0000.npu/cur_freq').read_text()),'temperatures_c':{(p/'type').read_text().strip():int((p/'temp').read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*')}}
def memory_sample(pid,detailed=False):
 result={'pid':pid}
 for name in (('status','smaps_rollup') if detailed else ('status',)):
  try:
   fields={}
   for line in (Path('/proc')/str(pid)/name).read_text().splitlines():
    key,sep,value=line.partition(':')
    if sep and (key.startswith(('Vm','Rss','Pss','Private_','Shared_')) or key in ('Anonymous','Swap','SwapPss','Locked')):
     fields[key]=value.strip()
   result[name]=fields
  except (FileNotFoundError,ProcessLookupError,PermissionError) as exc:result[name+'_error']=type(exc).__name__
 if not detailed:return result
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
started=time.monotonic();child=None;paused=False;report={'thermal_pause_threshold_c':70,'thermal_resume_threshold_c':63,'compile_scheduler_events':[],'stage':a.stage,'command':command,'status':'running','frequency_written':False,'expected_observed_npu_hz':a.expected_npu_hz,'timeout_seconds':a.timeout,'samples':[],'cleanup_errors':[],'monitor_sampling_policy':{'thermal_poll_interval_seconds':0.10,'status_interval_seconds':1.0,'detailed_smaps_interval_seconds':30.0},'monitor_version':'v126_CPU_only_sampled_hysteresis_observed_frequency_r1'}
try:
 initial=sample();report['samples'].append(initial);assert initial['npu_hz']==a.expected_npu_hz and max(initial['temperatures_c'].values())<75
 with log.open('x') as stream:
  child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  report['child_pid']=child.pid
  next_status=0.;next_detailed=0.
  while child.poll() is None:
   row=sample();elapsed=row['elapsed_seconds'];report['samples'].append(row)
   if row['npu_hz']!=a.expected_npu_hz or max(row['temperatures_c'].values())>=75 or row['elapsed_seconds']>a.timeout:raise RuntimeError('Clock/temperature/timeout observation limit')
   hot=max(row['temperatures_c'].values())
   if not paused and hot>=70:
    os.killpg(child.pid,signal.SIGSTOP);paused=True;report['compile_scheduler_events'].append({'event':'pause_compiler','elapsed_seconds':row['elapsed_seconds'],'temperature_c':hot})
   elif paused and hot<=63:
    os.killpg(child.pid,signal.SIGCONT);paused=False;report['compile_scheduler_events'].append({'event':'resume_compiler','elapsed_seconds':row['elapsed_seconds'],'temperature_c':hot})
   if elapsed>=next_status:
    detailed=elapsed>=next_detailed
    row['worker_memory']=memory_sample(child.pid,detailed=detailed)
    next_status=time.monotonic()-started+1.0
    if detailed:next_detailed=time.monotonic()-started+30.0
   time.sleep(.10)
  report['returncode']=child.returncode;report['status']='completed' if child.returncode==0 else 'failed_child'
except BaseException as exc:
 report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
finally:
 if child is not None:
  # Leader exit does not prove that its compiler workers have exited.
  saved_handlers={n:signal.signal(n,signal.SIG_IGN) for n in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT)}
  try:
   cleanup=cleanup_process_group(child)
   report['process_group_cleanup']=cleanup
   report['cleanup_errors'].extend(cleanup['errors'])
   if cleanup['leader_exited_with_live_members']:
    report['cleanup_errors'].append('Session leader exited while live compiler workers remained; not a completed stage')
   if cleanup['errors'] or cleanup['leader_exited_with_live_members']:report['status']='failed_cleanup'
  except BaseException as exc:
   report['cleanup_errors'].append('Process-group cleanup: '+f'{type(exc).__name__}: {exc}')
   report['status']='failed_cleanup'
  finally:
   for n,handler in saved_handlers.items():signal.signal(n,handler)
 if child is not None:report['returncode']=child.poll()
 try:report['samples'].append(sample())
 except BaseException as exc:report['cleanup_errors'].append('Final observation: '+str(exc))
 own=resource.getrusage(resource.RUSAGE_SELF);worker=resource.getrusage(resource.RUSAGE_CHILDREN)
 report['CPU_accounting']={'controller_user_seconds':own.ru_utime,'controller_system_seconds':own.ru_stime,'worker_process_tree_user_seconds':worker.ru_utime,'worker_process_tree_system_seconds':worker.ru_stime,'scope':'Whole process lifetime, including imports/loading/validation/hooks/I/O; not isolated forward benchmark.'}
 report['seconds']=time.monotonic()-started
 record.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report.get(k) for k in ('stage','status','returncode','seconds','error','cleanup_errors')}),flush=True)
raise SystemExit(0 if report['status']=='completed' and not report['cleanup_errors'] else 1)
