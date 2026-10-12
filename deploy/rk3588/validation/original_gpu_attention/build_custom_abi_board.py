from pathlib import Path
import ctypes as ct,hashlib,json,os,subprocess
E=Path('/mnt/vggt-data/vggt-slam2/experiments');S=E/'vggt-v83-source-r1';T=E/'vggt-v27-toolchain/root';O=E/'vggt-v83-custom-abi-r1';O.mkdir(exist_ok=False)
assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
compiler=T/'usr/bin/aarch64-linux-gnu-g++-11';assert sha(compiler)=='2aafdb1e153fc490fbd510d352572eee55ecdd29dd95eca239b4460c1afe3a12'
header=E.parent/'src/deploy/rk3588/reference/rknn_api_v2.3.2.h';assert sha(header)=='c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
(O/'rknn_api.h').symlink_to(header)
cmd=[str(compiler),'-x','c','-O2','-std=c11','-shared','-fPIC','--sysroot='+str(T),'-B'+str(T/'usr/bin'),'-B'+str(T/'usr/lib/gcc/aarch64-linux-gnu/11'),'-I'+str(S),'-I'+str(O),str(S/'probe_custom_op_abi.c'),'-o',str(O/'abi.so')]
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(T/'usr/lib/aarch64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH','')
r=subprocess.run(cmd,env=env,capture_output=True,text=True)
report=dict(command=cmd,returncode=r.returncode,output=r.stdout+r.stderr,compiler_sha256=sha(compiler),custom_header_sha256=sha(S/'rknn_custom_op.h'),api_header_sha256=sha(header),source_sha256=sha(S/'probe_custom_op_abi.c'))
if r.returncode==0:report['library_sha256']=sha(O/'abi.so')
(O/'build.json').write_text(json.dumps(report,indent=2)+'\n')
assert r.returncode==0,report
assert ct.CDLL(str(O/'abi.so')).main()==0
