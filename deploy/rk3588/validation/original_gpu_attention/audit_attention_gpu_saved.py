#!/usr/bin/env python3
"""Independently check saved original references/GPU outputs; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert sha(a.capture)=='3b4eb0b24d320699dd7d198fd3158454a3043ed7244a0290b267cbfc450e8fee'
    cap=json.loads(a.capture.read_text()); result=json.loads(a.probe.read_text())
    assert result['status']=='completed_component_not_quality_acceptance'
    assert result['GPU_calls']==28 and result['events_created']==result['events_released']==148
    assert result['GPU_governor_limits_unchanged'] and not result['cleanup_errors']
    rows=[]; count=0
    for role in result['cases']:
        case=next(c for c in cap['cases'] if c['tag']==role['case'])
        artifacts=case['roles'][role['role']]['artifacts']
        refs={}
        for key in ('proj_input','attention_tail','q_post_norm','k_post_norm','q_post_rope','k_post_rope'):
            record=artifacts[key]; path=a.capture.parent/record['file']
            assert sha(path)==record['sha256']
            refs[key]=np.load(path,allow_pickle=False)
            assert hashlib.sha256(refs[key].tobytes()).hexdigest()==record['raw_sha256']
        first=None; errors=[]
        for call in role['calls']:
            record=call['saved_output']; path=a.probe.parent/record['file']
            assert sha(path)==record['sha256'] and path.stat().st_size==record['bytes']
            with np.load(path,allow_pickle=False) as arrays:
                assert set(arrays.files)==set(record['arrays'])
                for key in arrays.files:
                    value=arrays[key]; expected=refs[key]
                    assert value.shape==expected.shape and value.dtype==expected.dtype==np.float32
                    assert np.isfinite(value).all()
                    assert hashlib.sha256(value.tobytes()).hexdigest()==record['arrays'][key]['raw_sha256']
                    delta=np.abs(value.astype(np.float64)-expected.astype(np.float64))
                    assert np.all(delta <= 1e-5+1e-4*np.abs(expected.astype(np.float64)))
                    errors.append(float(delta.max()));count+=1
                if first is None:first=arrays['proj_input'].copy()
                else:assert arrays['proj_input'].tobytes()==first.tobytes()
        assert len(role['calls'])==7
        metrics=role['metrics_after_close']
        assert metrics['fully_released'] and metrics['closed'] and not metrics['cleanup_errors']
        assert metrics['live_OpenCL_objects']==metrics['pending_owned_host_uploads']==metrics['live_owned_DMA_fds']==0
        durations=[c['elapsed_seconds'] for c in role['calls'][1:]]
        assert abs(np.mean(durations)-role['summary']['warm_complete_mean_seconds'])<1e-12
        rows.append(dict(case=role['case'],role=role['role'],saved_arrays_strict_pass=True,
            maximum_absolute_error=max(errors),warm_mean_seconds=float(np.mean(durations)),
            warm_repeats_byte_equal=True,all_resources_released=True))
    output=dict(schema='v83_saved_original_GPU_attention_audit',status='passed',
        actual_inference_executed=False,source_sha256=sha(__file__),capture_sha256=sha(a.capture),
        probe_sha256=sha(a.probe),saved_arrays_verified=count,strict_rtol=1e-4,strict_atol=1e-5,
        cases=rows,full_pair_or_full_model_speedup_validated=False)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__':main()
