#!/usr/bin/env python3
"""Pure frozen receipt validation after reversible storage relocation; no SDK."""
import hashlib,importlib.util,json,time
from pathlib import Path
s=Path('/mnt/vggt-data/vggt-slam2/experiments/vggt-v100-source-r1')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
p=s/'full-native-sequential-config-r1.json'
assert sha(p)=='7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
c=json.loads(p.read_text()); source=s/'run_native_retained_groups.py'
assert sha(source)==c['sources'][source.name]
spec=importlib.util.spec_from_file_location('_storage_check',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
t=time.monotonic(); rows,lineage=m.load_compilations([x['file'] for x in c['aggregator']],[x['sha256'] for x in c['aggregator']],s/c['aggregator_compiler'],c['sources'][c['aggregator_compiler']])
r=dict(status='verified_all_four_original_compilation_lineages',SDK_or_model_executed=False,seconds=time.monotonic()-t,models=rows,lineage=lineage)
print(json.dumps(r,indent=2))
