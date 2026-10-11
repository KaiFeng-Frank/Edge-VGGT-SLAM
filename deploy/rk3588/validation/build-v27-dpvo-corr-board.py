#!/usr/bin/env python3
"""Build the isolated native research library on RK3588, then tiny smoke."""
from pathlib import Path
import argparse,platform,json,hashlib,subprocess,os,importlib.util,traceback
BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
assert platform.machine()=='aarch64' and 'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes().decode()
out=a.output_dir.resolve();assert out.is_relative_to(BASE);out.mkdir(exist_ok=False)
source_pins={'dpvo_corr.cpp':'3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990','dpvo_corr.h':'1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932','dpvo_corr.py':'a6c37bf4b840b492736d5c5916be6e2c0e8680693d01d0fe3a5fe62702ecb779'}
r=dict(status='failed',script_sha256=sha(__file__),source_sha256=source_pins)
try:
 for name,digest in source_pins.items():assert sha(a.source_dir/name)==digest
 T=BASE/'vggt-v27-toolchain';R=T/'root';prior=json.loads((T/'bootstrap-r2.json').read_text());assert prior['status']=='isolated_compiler_and_synthetic_smoke_passed'
 compiler=R/'usr/bin/aarch64-linux-gnu-g++-11';assert sha(compiler)==prior['compiler_sha256']
 env=os.environ.copy();env['LD_LIBRARY_PATH']=str(R/'usr/lib/aarch64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
 target=out/'libdpvo_corr.so'
 cmd=[str(compiler),'-O3','-std=c++17','-shared','-fPIC','-fopenmp','-fno-fast-math','-ffp-contract=off','--sysroot='+str(R),'-B'+str(R/'usr/bin'),'-B'+str(R/'usr/lib/gcc/aarch64-linux-gnu/11'),str(a.source_dir/'dpvo_corr.cpp'),'-o',str(target)]
 d=subprocess.run(cmd,env=env,capture_output=True,text=True);r.update(command=cmd,exit_code=d.returncode,output_log=d.stdout+d.stderr,compiler_sha256=sha(compiler),toolchain_bootstrap_sha256=sha(T/'bootstrap-r2.json'),toolchain_manifest_sha256=sha(T/'manifest.json'))
 assert d.returncode==0,d.stderr
 import numpy as np
 spec=importlib.util.spec_from_file_location('_v27_native_smoke',a.source_dir/'dpvo_corr.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 g=np.ones((1,1,8,1,1),np.float32);f=np.ones((1,1,8,8,8),np.float32);xy=np.full((1,1,2,1,1),3.0,np.float32);idx=np.zeros(1,np.int64)
 with m.PreparedCorrelation(target,g,f,threads=4,cache_key='tiny-smoke',library_sha256=sha(target)) as prep:
  corr,timing=prep.correlate(xy,idx,idx,cache_key='tiny-smoke');value=corr.reshape(1,1,7,7,1,1,2)[0,0,3,3,0,0]
  assert np.array_equal(value,np.array([8,8],np.float32));metrics=prep.metrics()
 final=prep.metrics();assert final['closed'] and final['current_context_bytes']==0
 r.update(status='compiled_and_synthetic_smoke_passed',smoke=dict(center=value.tolist(),before_close=metrics,after_close=final),output=dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target)))
 r['source_sha256_after']={name:sha(a.source_dir/name) for name in source_pins};assert r['source_sha256_after']==source_pins
except Exception as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
(out/'build.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:r.get(k) for k in ('status','error','output')}),flush=True)
raise SystemExit(0 if r['status']=='compiled_and_synthetic_smoke_passed' else 1)
