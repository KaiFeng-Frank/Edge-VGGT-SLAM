import argparse,ctypes as ct,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes();assert hashlib.sha256(a.library.read_bytes()).hexdigest()==a.sha256
lib=ct.CDLL(str(a.library));lib.f32_to_f16.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];lib.f32_to_f16.restype=None;lib.f16_to_f32.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t];lib.f16_to_f32.restype=None
rows=[]
# Every finite half value, including signs, zeros and subnormals, and odd offsets/tails.
h=np.arange(65536,dtype=np.uint16).view(np.float16);h=h[np.isfinite(h)]
f=np.empty(h.shape,np.float32);lib.f16_to_f32(h.ctypes.data,f.ctypes.data,h.size);assert f.tobytes()==h.astype(np.float32).tobytes();rows.append('every finite half encoding widened exactly')
rng=np.random.default_rng(56);v=rng.integers(0,2**32,size=300000,dtype=np.uint32).view(np.float32);v=v[np.isfinite(v)&(np.abs(v)<=65504)]
for n in [0,1,2,3,4,5,7,8,9,15,16,17,len(v)-1]:
 f=np.empty(n+2,np.float32);f[:]=1.25;f[1:1+n]=v[:n]
 out=np.full(n+2,0x55aa,np.uint16);lib.f32_to_f16(f[1:].ctypes.data,out[1:].ctypes.data,n)
 assert out[0]==out[-1]==0x55aa
 assert out[1:1+n].tobytes()==v[:n].astype(np.float16).view(np.uint16).tobytes()
 back=np.full(n+2,13.5,np.float32);lib.f16_to_f32(out[1:].ctypes.data,back[1:].ctypes.data,n)
 assert back[0]==back[-1]==13.5
 assert back[1:1+n].tobytes()==out[1:1+n].view(np.float16).astype(np.float32).tobytes()
 rows.append(dict(count=n,unaligned_to_vector=True,guard_elements_preserved=True,bitwise_equal=True))
r=dict(status='passed_generated_finite_conversion_checks',library_sha256=a.sha256,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows,model_or_SDK_executed=False)
with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(status=r['status'],cases=len(rows))))
