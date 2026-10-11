#!/usr/bin/env python3
"""Isolated RK3588 native pack build; no model loading or execution."""
import argparse,ctypes,hashlib,json,os,platform,subprocess,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
BOOTSTRAP='868c1aaf324f6263db743c19d62e493cc643646a0e2035f70b2efcb08448093e'
MANIFEST='5b7c9f15b2af1975fee0de57e78c9176ecda113aae1ec1bf48a7453f05ead5d6'
GXX='2aafdb1e153fc490fbd510d352572eee55ecdd29dd95eca239b4460c1afe3a12'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,m):
 if not x:raise ValueError(m)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('source-dir','toolchain-dir','output-dir','source-pins'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--source-pins-sha256',required=True)
 a=p.parse_args()
 require(platform.machine()=='aarch64' and platform.system()=='Linux','RK3588 board only')
 require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board only')
 out=a.output_dir.resolve();require(out.is_relative_to(BASE),'Isolated experiment required');out.mkdir(exist_ok=False)
 record=dict(schema='v43_native_fp16_pack_isolated_build',status='failed',candidate_accepted=False,model_executed=False,cleanup_errors=[])
 pins={}
 def pin(p,e):
  p=Path(p).resolve(strict=True);require(sha(p)==e,'Source SHA differs: '+str(p));pins[str(p)]=e;return p
 try:
  pin(__file__,sha(__file__))
  frozen=json.loads(pin(a.source_pins,a.source_pins_sha256).read_text())
  for n,d in frozen.items():pin(a.source_dir/n,d)
  t=a.toolchain_dir.resolve();r=t/'root'
  bootstrap=json.loads(pin(t/'bootstrap-r2.json',BOOTSTRAP).read_text())
  require(bootstrap['status']=='isolated_compiler_and_synthetic_smoke_passed','Unverified toolchain')
  pin(t/'manifest.json',MANIFEST);pin(r/'usr/bin/aarch64-linux-gnu-g++-11',GXX)
  compiler=pin(r/'usr/bin/aarch64-linux-gnu-gcc-11',sha(r/'usr/bin/aarch64-linux-gnu-gcc-11'))
  output=out/'libv43_native_fp16_pack.so'
  command=[str(compiler),'-O3','-std=c11','-shared','-fPIC','-march=armv8.2-a+fp16','-fno-fast-math','-ffp-contract=off','--sysroot='+str(r),'-B'+str(r/'usr/bin'),'-B'+str(r/'usr/lib/gcc/aarch64-linux-gnu/11'),str(a.source_dir/'v43_native_fp16_pack.c'),'-o',str(output)]
  env=os.environ.copy();env['LD_LIBRARY_PATH']=str(r/'usr/lib/aarch64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','');env['TMPDIR']=str(out)
  result=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True)
  record.update(command=command,compiler_sha256=sha(compiler),compiler_pinning='measured_before_build_with_frozen_existing_toolchain_manifest',source_sha256=frozen,exit_code=result.returncode,output_log=result.stdout+result.stderr)
  require(result.returncode==0,'Native compile failed')
  lib=ctypes.CDLL(str(output));lib.v43_pack_abi_version.restype=ctypes.c_uint32;lib.v43_pack_read_fpcr.restype=ctypes.c_uint64
  require(lib.v43_pack_abi_version()==1,'ABI differs')
  record.update(status='compiled_ABI_loaded_conversion_not_yet_verified',ABI=1,FPCR=int(lib.v43_pack_read_fpcr()),output=dict(file=output.name,bytes=output.stat().st_size,sha256=sha(output)))
 except Exception as error:record.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  try:
   after={p:sha(p) for p in pins};record.update(source_pins=pins,source_hashes_after=after,sources_unchanged=after==pins);require(after==pins,'Source changed')
  except Exception as error:record['cleanup_errors'].append(repr(error));record['status']='failed'
  (out/'build.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
 print(json.dumps(record,indent=2,allow_nan=False))
 return 0 if record['status']=='compiled_ABI_loaded_conversion_not_yet_verified' else 1
if __name__=='__main__':raise SystemExit(main())
