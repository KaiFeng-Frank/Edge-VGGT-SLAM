"""Saved receipts and arrays only. Does not decode images or execute a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--run-dir',type=Path,required=True)
p.add_argument('--run-sha256',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(a.run_dir/'run.json')==a.run_sha256
x=json.loads((a.run_dir/'run.json').read_text())
assert x['status']=='completed_preprocessing_pixel_gate_passed_not_realtime'
assert x['frames_completed']==192 and x['all192_pixel_gate_passed']
assert x['OpenCV_version']=='4.10.0' and x['OpenCV_threads_both_paths']==4
assert x['affinity_restored'] and x['OpenCV_threads_restored']
assert x['sources_and_inputs_unchanged'] and x['source_pins']==x['source_hashes_after']
assert not x['cleanup_errors'] and x['cache']['closed']
assert all(x[k]==0 for k in ('neural_forward_calls','ORT_calls','SDK_calls','BA_calls'))
assert [r['index'] for r in x['frames']]==list(range(192))
for row in x['frames']:
    assert row['bitwise_all_three'] and row['original_arrays']==row['cached_arrays']
    assert [r['shape'] for r in row['original_arrays']]==[[480,640,3],[232,304,3],[1,3,232,304]]
    assert row['imread_seconds']>=0
selected=[i*191//23 for i in range(24)]
assert x['performance_source_indices']==selected
assert [(r['frame_index'],r['method']) for r in x['performance_calls']]==[(i,m) for i in selected for m in ('original','cached','cached','original')]
timing_keys=('undistort_or_cached_remap_seconds','crop_resize_seconds','normalization_seconds','CPU_prep_without_decode_seconds')
for r in x['performance_calls']:
    assert r['bitwise_all_three'] and all(np.isfinite(r[k]) and r[k]>=0 for k in timing_keys)
    assert abs(sum(r[k] for k in timing_keys[:3])-r[timing_keys[3]])<1e-8
sample=x['samples'];file=a.run_dir/sample['file']
assert file.stat().st_size==sample['bytes'] and sha(file)==sample['sha256']
with np.load(file,allow_pickle=False) as z:
    arrays={k:z[k] for k in z.files}
assert set(arrays)==set(sample['arrays']) and len(arrays)==12
for k,v in arrays.items():
    assert dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes,tensor_sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest())==sample['arrays'][k]
for index in (0,95,191):
    for kind in ('half','normalized'):
        assert arrays[f'original_{kind}_{index:03d}'].tobytes()==arrays[f'cached_{kind}_{index:03d}'].tobytes()
stats={}
for method in ('original','cached'):
    rows=[r for r in x['performance_calls'] if r['method']==method]
    stats[method]={k:dict(count=len(rows),mean_ms=float(np.mean([r[k] for r in rows])*1000),p95_ms=float(np.percentile([r[k] for r in rows],95)*1000)) for k in timing_keys}
report=dict(schema='v40_saved_preprocessing_audit',status='passed',run_sha256=a.run_sha256,
    metadata_pixel_frames_checked=192,actual_sample_arrays_reread=12,actual_sample_pairs_compared=6,
    ABBA_performance_calls_checked=96,statistics=stats,cold_map_construction_ms=x['cold_map_construction_seconds']*1000,
    map_bytes=x['cache']['map_bytes'],RSS_MiB=x['peak_rss_mib'],
    scope='Board reports all192 byte comparisons; host verifies those metadata and six saved actual array pairs for three images. No host real-image preprocessing or inference.',
    no_end_to_end_replay=True,realtime_validated=False)
assert not a.output.exists();a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
