import hashlib,json,os,subprocess
from pathlib import Path
assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
E=Path('/mnt/vggt-data/vggt-slam2/experiments');T=E/'vggt-v27-toolchain';R=T/'root';S=E/'vggt-v56-source-r1';O=E/'vggt-v56-neon-build-r1';O.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
compiler=R/'usr/bin/aarch64-linux-gnu-g++-11';assert sha(compiler)=='2aafdb1e153fc490fbd510d352572eee55ecdd29dd95eca239b4460c1afe3a12'
cmd=[str(compiler),'-x','c','-O3','-std=c11','-shared','-fPIC','-march=armv8.2-a+fp16','-ffp-contract=off','-fno-fast-math','--sysroot='+str(R),'-B'+str(R/'usr/bin'),'-B'+str(R/'usr/lib/gcc/aarch64-linux-gnu/11'),str(S/'fp16_neon.c'),'-o',str(O/'fp16_neon.so')]
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(R/'usr/lib/aarch64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
r=subprocess.run(cmd,env=env,capture_output=True,text=True);report=dict(status='compiled' if r.returncode==0 else 'failed',command=cmd,exit_code=r.returncode,output=r.stdout+r.stderr,compiler_sha256=sha(compiler),source_sha256=sha(S/'fp16_neon.c'))
if r.returncode==0:report['library_sha256']=sha(O/'fp16_neon.so')
(O/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(r.returncode)
