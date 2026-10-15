#!/usr/bin/env python3
"""Reversibly move unchanged private export storage off tmpfs; no model/SDK calls."""
import hashlib,json,os,shutil
from pathlib import Path
src=Path('/dev/shm/edge-vggt-v100-native/agg-group0-export-r1')
dst=Path('/tmp/vggt-v123-export-storage/agg-group0-export-r1')
report=Path('/tmp/vggt-v123-export-storage/relocation.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def zones():
 out={};key=None
 for s in Path('/proc/zoneinfo').read_text().splitlines():
  if s.startswith('Node '):key=s.strip()
  elif key and s.strip().startswith('pages free'):out[key]=int(s.split()[-1])*os.sysconf('SC_PAGE_SIZE')
 return out
assert src.is_dir() and not src.is_symlink() and not dst.exists()
files=sorted(src.iterdir());assert len(files)==2 and {p.name for p in files}=={'export.json','retained-pair-group-00-04-native518.onnx'}
assert all(p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 for p in files)
rows=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in files]
assert sum(r['bytes'] for r in rows)==512156018
assert shutil.disk_usage('/tmp').free>sum(r['bytes'] for r in rows)+(128<<20)
record=dict(schema='v123_reversible_original_export_storage_relocation',source=str(src),destination=str(dst),files=rows,before_zone_free_bytes=zones(),model_or_SDK_execution=False,driver_clock_governor_or_memory_policy_changed=False)
dst.parent.mkdir(parents=True,exist_ok=True)
shutil.move(str(src),str(dst))
try:
 for r in rows:assert (dst/r['file']).stat().st_size==r['bytes'] and sha(dst/r['file'])==r['sha256']
 src.symlink_to(dst,target_is_directory=True)
 for r in rows:assert sha(src/r['file'])==r['sha256']
 record.update(status='moved_verified_original_path_directory_symlink',after_verified_reads_zone_free_bytes=zones(),disk_free_bytes=shutil.disk_usage('/tmp').free,all_original_bytes_preserved=True,rollback='Remove only the source directory symlink, then move the complete destination directory back to its original source path; verify the listed hashes.')
except BaseException:
 if not src.exists() and dst.exists():shutil.move(str(dst),str(src))
 raise
report.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
