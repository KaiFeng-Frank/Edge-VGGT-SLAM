"""Saved v39 metadata only: catalog coverage of complete active Update graphs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(a.run)=='dd9f37524b999c1e586182a92c226481fcfc7f4da851d5e88d62d82be1029ecd'
assert sha(a.manifest)=='5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
run=json.loads(a.run.read_text()); manifest=json.loads(a.manifest.read_text())
full=[x for x in run['update_calls'] if x['stage']=='stream' and x['edges']>16]
actual=Counter(); bounded=Counter(); examples={}
for call in full:
    assert len(call['source_degrees'])==len(call['source_slots'])==len(call['source_patch_counts'])
    assert [d*16 for d in call['source_degrees']]==call['source_patch_counts']
    assert sum(call['source_patch_counts'])==call['edges']
    for degree,batch in Counter(call['source_degrees']).items():
        key=f'B{batch}_D{degree}_M16';actual[key]+=1
        examples.setdefault(key,dict(observation=call['frame_index'],update_index=call['index'],edges=call['edges']))
        pairs,remainder=divmod(batch,2)
        if pairs:bounded[f'B2_D{degree}_M16']+=pairs
        if remainder:bounded[f'B1_D{degree}_M16']+=1
catalogue=set(manifest['models'])
result=dict(schema='v40_saved_M16_shape_coverage',status='saved_metadata_analyzed',
    original_run_sha256=sha(a.run),export_manifest_sha256=sha(a.manifest),
    full_online_calls=len(full),includes_initialization=True,excludes_EOF_and_motion_probes=True,
    actual_groups=dict(sorted(actual.items())),first_examples=examples,captured_catalogue=sorted(catalogue),
    missing_actual_shapes=sorted(set(actual)-catalogue),
    source_batch_at_most_two=dict(sorted(bounded.items())),
    missing_shapes_after_complete_source_splitting=sorted(set(bounded)-catalogue),
    total_current_groups=sum(actual.values()),total_groups_with_max_source_batch_two=sum(bounded.values()),
    complete_source_split_has_been_executed_on_NPU=False,
    new_shapes_exported=False,model_inference_executed=False,candidate_accepted=False,
    limitations=['Saved original CPU trajectory only; changed recurrent predictions may change graph shapes.',
      'Complete source-axis splits preserve D*16 dependencies; never split D or patch reductions.',
      'More SDK groups need measured total packing/transfer/cache time.',
      'Motion probe degree one requires a separate explicit route.',
      'Unseen shapes must fail closed; current three-state diagnostic is not full online coverage.'])
assert not a.output.exists()
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('full_online_calls','missing_actual_shapes','missing_shapes_after_complete_source_splitting','total_current_groups','total_groups_with_max_source_batch_two')},indent=2))
