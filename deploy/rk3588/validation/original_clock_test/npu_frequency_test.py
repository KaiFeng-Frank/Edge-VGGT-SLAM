#!/usr/bin/env python3
"""Bounded original-VGGT test with an explicitly authorized NPU frequency change.

No CPU governor, voltage, cooling, driver, network or model modification.
Restores the original actual NPU frequency and verifies the governor in finally. An initially
undefined userspace request becomes an explicit request for that same frequency;
this is recorded, never described as byte-identical sysfs state restoration.
"""
import argparse,json,os,signal,subprocess,time,traceback
from pathlib import Path

DEV=Path('/sys/class/devfreq/fdab0000.npu')
THERM=Path('/sys/class/thermal')

def read(name):return (DEV/name).read_text().strip()
def temps(zones):
 # Keep every baseline node in the monitoring set. A vanished/unreadable node
 # must fail the run, rather than silently making the board look cooler.
 return {p.joinpath('type').read_text().strip():int(p.joinpath('temp').read_text())/1000 for p in zones}
def record(path,data):
 q=path.with_suffix('.tmp');q.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');q.replace(path)
def stop(child,pgid):
 # start_new_session makes the saved child PID the process-group ID, even
 # after the launcher exits. Do not mistake a reaped leader for an empty group.
 if pgid is None:
  if child is None:return
  pgid=child.pid
 def group_alive():
  if child is not None:child.poll() # Reap our leader before checking the group.
  try:os.killpg(pgid,0);return True
  except ProcessLookupError:return False
 for sig in (signal.SIGTERM,signal.SIGKILL):
  try:os.killpg(pgid,sig)
  except ProcessLookupError:
   if child is not None:child.poll()
   return
  deadline=time.monotonic()+5
  while group_alive():
   if time.monotonic()>=deadline:break
   time.sleep(.05)
  else:return
 raise RuntimeError(f'Worker process group {pgid} still exists after SIGKILL')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--execute-frequency-test',action='store_true',required=True)
 p.add_argument('--frequency-hz',type=int,default=1000000000)
 p.add_argument('--temperature-limit-c',type=float,default=75.)
 p.add_argument('--timeout-seconds',type=float,default=180.)
 p.add_argument('--launcher',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():p.error('RK3588 board required')
 if not 0<a.timeout_seconds<=180 or not 45<=a.temperature_limit_c<=75:p.error('Bounded timeout <=180s and temperature <=75C required')
 available=list(map(int,read('available_frequencies').split()))
 if a.frequency_hz not in available:p.error('Only a kernel-advertised frequency is permitted')
 a.output.mkdir(parents=True,exist_ok=False)
 old={k:read(k) for k in ['governor','cur_freq','min_freq','max_freq','userspace/set_freq']}
 if old['governor']!='userspace':p.error('Expected existing userspace governor; no governor changes allowed by this launcher')
 if not int(old['min_freq'])<=a.frequency_hz<=int(old['max_freq']):p.error('Requested frequency outside existing limits')
 thermal_zones=tuple(sorted(THERM.glob('thermal_zone*')))
 if not thermal_zones:p.error('Thermal monitoring nodes required')
 baseline_temps=temps(thermal_zones)
 if max(baseline_temps.values())>=a.temperature_limit_c:p.error('Temperature already above stop threshold')
 report=dict(status='starting',original_model_only=True,baseline=old,requested_frequency_hz=a.frequency_hz,temperature_limit_c=a.temperature_limit_c,timeout_seconds=a.timeout_seconds,baseline_temperatures_c=baseline_temps,observations=[],child_returncode=None,restore_errors=[],candidate_accepted=False)
 child=None;pgid=None;changed=False;begin=time.monotonic()
 def interrupted(signum,frame):raise RuntimeError('Interrupted by signal '+str(signum))
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,interrupted)
 try:
  record(a.output/'control.json',report)
  changed=True;(DEV/'userspace/set_freq').write_text(str(a.frequency_hz))
  actual=int(read('cur_freq'))
  if actual!=a.frequency_hz:raise RuntimeError(f'Frequency request not realized: {actual}')
  with (a.output/'worker.log').open('wb') as log:
   child=subprocess.Popen(['sh',str(a.launcher.resolve(strict=True))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   pgid=child.pid
   while True:
    state=dict(seconds=time.monotonic()-begin,actual_frequency_hz=int(read('cur_freq')),temperatures_c=temps(thermal_zones))
    report['observations'].append(state);record(a.output/'control.json',report)
    if max(state['temperatures_c'].values())>=a.temperature_limit_c:raise RuntimeError('Thermal stop threshold reached')
    if time.monotonic()-begin>a.timeout_seconds:raise RuntimeError('Bounded test timeout reached')
    code=child.poll()
    if code is not None:
     report['child_returncode']=code
     if code:raise RuntimeError('Worker failed: '+str(code))
     report['status']='worker_completed_pending_result_review';break
    time.sleep(.5)
 except BaseException as exc:
  report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
 finally:
  # Once cleanup starts, repeated termination signals must not abort stopping
  # the worker group or restoring the original actual NPU frequency.
  for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,signal.SIG_IGN)
  try:stop(child,pgid)
  except BaseException as exc:report['restore_errors'].append('worker stop: '+str(exc))
  if child is not None:report['child_returncode']=child.poll()
  if changed:
   try:
    (DEV/'userspace/set_freq').write_text(old['cur_freq'])
    restored={k:read(k) for k in old};report['restored']=restored
    report['userspace_request_was_undefined']=old['userspace/set_freq']=='undefined'
    if restored['cur_freq']!=old['cur_freq'] or restored['governor']!=old['governor'] or restored['min_freq']!=old['min_freq'] or restored['max_freq']!=old['max_freq']:raise RuntimeError('Original frequency/governor/limits not restored')
    report['frequency_governor_limits_restored']=True
   except BaseException as exc:report['restore_errors'].append('NPU restore: '+str(exc))
  report['seconds']=time.monotonic()-begin
  if report['restore_errors']:report['status']='failed_restore'
  record(a.output/'control.json',report)
 print(json.dumps({k:report.get(k) for k in ['status','child_returncode','seconds','frequency_governor_limits_restored','restore_errors']}),flush=True)
 return 0 if report['status']=='worker_completed_pending_result_review' else 1
if __name__=='__main__':raise SystemExit(main())
