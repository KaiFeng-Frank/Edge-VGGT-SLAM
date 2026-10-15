#!/usr/bin/env python3
"""Read-only resident-page metadata for existing board artifacts; no model execution."""
import json,mmap,os,struct
from pathlib import Path
ROOT=Path('/dev/shm/edge-vggt-v100-native')
paths=[ROOT/'agg-group0-export-r1/retained-pair-group-00-04-native518.onnx',ROOT/'agg-group1-export-r1/retained-pair-group-05-11-native518.onnx',ROOT/'agg-group1-hostcompile-r1/group.rknn']
rows=[];page=os.sysconf('SC_PAGE_SIZE')
for path in paths:
 with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as view:
  touched=view[::page]
  del touched
  maps=[x for x in Path('/proc/self/maps').read_text().splitlines() if x.split(maxsplit=5)[-1]==str(path.resolve())]
  assert len(maps)==1,maps
  address=int(maps[0].split('-')[0],16);count=(path.stat().st_size+page-1)//page
  with open('/proc/self/pagemap','rb',buffering=0) as pm:
   pm.seek(address//page*8);entries=pm.read(count*8)
  assert len(entries)==count*8
  present=low=masked=swapped=0
  for (entry,) in struct.iter_unpack('Q',entries):
   if entry>>63&1:
    present+=1;pfn=entry&((1<<55)-1)
    if pfn==0:masked+=1
    elif pfn*page<1<<32:low+=1
   if entry>>62&1:swapped+=1
  rows.append(dict(file=str(path),bytes=path.stat().st_size,pages=count,present_pages=present,zero_or_restricted_PFN_pages=masked,physical_below_4GiB_bytes=low*page,swapped_pages=swapped))
print(json.dumps(dict(schema='read_only_artifact_page_residency_v123',page_size=page,rows=rows),indent=2))
