#!/usr/bin/env python3
"""Audit saved dense encoder arrays only; no Torch/ORT/SDK or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def meta(x):
    return dict(shape=list(x.shape),dtype=str(x.dtype),bytes=x.nbytes,
                tensor_sha256=hashlib.sha256(x.tobytes(order='C')).hexdigest())


def require(ok,why):
    if not ok:raise ValueError(why)


def read(folder,entry,metadata):
    name=entry['file'];require(Path(name).name==name,'Unsafe array basename')
    p=folder/name;require(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'Saved NPZ identity differs')
    with np.load(p,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    require(set(arrays)=={'fmap','imap'} and {k:meta(v) for k,v in arrays.items()}==metadata,'Saved array identity differs')
    return arrays


def audit(run_dir,capture_dir,run_sha,capture_sha):
    rp=run_dir/'run.json';cp=capture_dir/'capture.json'
    require(sha(rp)==run_sha and sha(cp)==capture_sha,'Receipt SHA differs')
    run=json.loads(rp.read_text());cap=json.loads(cp.read_text())
    require(run['schema']=='v35_DPVO_original_FAST_dense_spatial_RKNN_encoder_diagnostic_replay','Wrong replay schema')
    require(run['frames_completed']==192 and run['original_weights_unchanged'] and not run['cleanup_errors'],'Complete unchanged-weight replay required')
    require(run['first_three_FP32_gate_passed'] is False and run['candidate_accepted'] is False,'Historical rejection must remain')
    rows=[]
    for frame in run['frames'][:3]:
        i=frame['index'];reference=next(c for c in cap['calls'] if c['frame_index']==i)
        require(frame['frame_id']==reference['frame_id'] and frame['original_input_raw_bytes_match'],'Exact input identity required')
        entry=frame['first_three_dense_encoder_artifact'];actual=read(run_dir,entry,entry['arrays'])
        expected=read(capture_dir,reference['artifact'],reference['arrays'])
        for key in actual:
            x,y=actual[key],expected[key]
            require(x.shape==y.shape and x.dtype==y.dtype==np.float32,'Shape/dtype mismatch')
            require(np.isfinite(x).all() and np.isfinite(y).all(),'Nonfinite saved output')
            d=np.abs(x.astype(np.float64)-y.astype(np.float64));limit=1e-5+1e-4*np.abs(y.astype(np.float64))
            failed=int(np.count_nonzero(d>limit));maximum=float(d.max())
            n=float(np.linalg.norm(y.astype(np.float64).ravel()));relative=float(np.linalg.norm(d.ravel())/n) if n else None
            logged=frame['first_three_dense_encoder_gate'][key]
            require(logged['rtol']==1e-4 and logged['atol']==1e-5 and logged['failed_elements']==failed and logged['element_count']==x.size and logged['gate_passed']==(failed==0),'Saved numerical gate differs')
            require(logged['max_abs']==maximum,'Saved maximum error differs')
            rows.append(dict(frame_index=i,array=key,elements=x.size,failed_elements=failed,gate_passed=failed==0,max_abs=maximum,relative_l2=relative,board_reported_relative_l2=logged['relative_l2'],relative_l2_summary_absolute_difference=abs(relative-logged['relative_l2']),actual_sha256=entry['sha256'],reference_sha256=reference['artifact']['sha256']))
    require(sha(rp)==run_sha and sha(cp)==capture_sha,'Receipt changed while auditing')
    return dict(schema='v35_FAST_dense_encoder_saved_array_audit',status='passed_evidence_consistent_numerical_rejection_retained',script_sha256=sha(__file__),run_sha256=run_sha,capture_sha256=capture_sha,comparisons=len(rows),failed_evidence_checks=0,original_FP32_gate_passed=all(r['gate_passed'] for r in rows),candidate_accepted=False,relative_L2_is_descriptive_not_acceptance_gate=True,relative_L2_reduction_bit_parity_claimed=False,actual_model_executed=False,ORT_executed=False,SDK_executed=False,rows=rows,scope='First three full dense fmap/imap saved arrays, unchanged FP32 thresholds and raw identities. Patch selection and complete recurrent trajectory are separately reported; no additional inference.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('run-dir','capture-dir','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('run-sha256','capture-sha256'):p.add_argument('--'+n,required=True)
    a=p.parse_args();require(not a.output.exists(),'New output required')
    d=audit(a.run_dir,a.capture_dir,a.run_sha256,a.capture_sha256)
    with a.output.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:d[k] for k in ('status','comparisons','original_FP32_gate_passed')}))


if __name__=='__main__':main()
