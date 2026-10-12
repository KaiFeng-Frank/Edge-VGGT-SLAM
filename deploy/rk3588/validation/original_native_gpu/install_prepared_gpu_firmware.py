#!/usr/bin/env python3
"""Install one absent, pinned GPU firmware file only after explicit authorization."""
import argparse, hashlib, json, os
from pathlib import Path
EXPECTED='7bd66578c787a9f0e373ba4e9eb3db309bcde91e5451be5ee12e2e90007e60f6'
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--execute-authorized-install',action='store_true',required=True)
 p.add_argument('--source',type=Path,required=True)
 p.add_argument('--receipt',type=Path,required=True)
 a=p.parse_args()
 assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
 assert Path('/sys/module/bifrost_kbase/version').read_text().startswith('g11p0-01eac0')
 raw=a.source.read_bytes();assert len(raw)==266240 and hashlib.sha256(raw).hexdigest()==EXPECTED
 target=Path('/lib/firmware/mali_csffw.bin'); stage=target.with_name('.mali_csffw.v81-staging')
 assert target.parent.is_dir() and not target.exists() and not target.is_symlink()
 assert not stage.exists() and not stage.is_symlink() and not a.receipt.exists()
 prior={str(x):x.read_text().strip() for x in [Path('/sys/class/devfreq/fb000000.gpu/cur_freq'),Path('/sys/class/devfreq/fb000000.gpu/governor')]}
 with stage.open('xb') as f:
  f.write(raw);f.flush();os.fsync(f.fileno())
 os.chmod(stage,0o644)
 try:
  os.link(stage,target)
 finally:
  stage.unlink()
 assert hashlib.sha256(target.read_bytes()).hexdigest()==EXPECTED
 receipt={'status':'installed_absent_matching_g11p0_GPU_firmware','source':str(a.source),'target':str(target),'bytes':len(raw),'sha256':EXPECTED,'previous_target_absent':True,'driver_kernel_governor_or_frequency_writes':False,'existing_GPU_observation':prior,'GPU_context_or_model_started_by_installer':False}
 a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
