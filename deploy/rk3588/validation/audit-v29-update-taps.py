#!/usr/bin/env python3
"""NumPy-only saved-output comparison; never executes ONNX or a model."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ORDER=['/corr/corr.2/Add_output_0','/corr/corr.3/LayerNormalization_output_0',
       '/norm/LayerNormalization_output_0','/Add_3_output_0','/Gather_2_output_0',
       '/Where_output_0','/ReduceMax_output_0','/Exp_output_0','/Div_output_0',
       '/Add_5_output_0','/Gather_5_output_0','/Where_3_output_0','/Div_1_output_0',
       '/Add_7_output_0','/gru/gru.0/LayerNormalization_output_0','/gru/gru.2/LayerNormalization_output_0']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(x):return dict(shape=list(x.shape),dtype=str(x.dtype),bytes=x.nbytes,tensor_sha256=hashlib.sha256(x.tobytes(order='C')).hexdigest())
def counts(x):
    finite=np.isfinite(x);good=x[finite]
    return dict(elements=x.size,finite=int(finite.sum()),nan=int(np.isnan(x).sum()),positive_inf=int(np.isposinf(x).sum()),negative_inf=int(np.isneginf(x).sum()),all_finite=bool(finite.all()),all_zero=bool((x==0).all()),finite_min=float(good.min()) if good.size else None,finite_max=float(good.max()) if good.size else None)
def arrays(folder,record,pins):
    item=record['outputs'];name=item['file'];assert Path(name).name==name
    path=folder/name;assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes'];pins[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    assert {k:meta(v) for k,v in data.items()}==item['arrays']
    assert {k:counts(v) for k,v in data.items()}==record['finite_counts']
    return data
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ort-dir','sdk-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();pins={};runs=[];values=[]
    for folder in (a.ort_dir,a.sdk_dir):
        path=folder/'run.json';pins[str(path)]=sha(path);r=json.loads(path.read_text())
        assert r['execution_status']=='completed' and not r['cleanup_errors'] and r['sources_and_inputs_unchanged'] and r['affinity_restored'] and len(r['calls'])==1
        runs.append(r);values.append(arrays(folder,r['calls'][0],pins))
    ort,sdk=runs;reference,actual=values
    assert ort['mode']=='ort' and sdk['mode']=='sdk-profile' and sdk['destroy_code']==0
    assert ort['manifest_sha256']==sdk['manifest_sha256'] and sdk['calls'][0]['timing']['outputs_released']
    assert all(ort['calls'][0][k]==sdk['calls'][0][k] for k in ('case','step','model_key'))
    assert set(actual)==set(reference)==set(ORDER+['net_out','delta','weight'])
    rows=[]
    for key in ORDER+['net_out','delta','weight']:
        x,y=actual[key],reference[key];assert x.dtype==y.dtype==np.float32 and x.shape==y.shape
        cf,rf=counts(x),counts(y);row=dict(name=key,SDK=cf,ORT=rf)
        if cf['all_finite'] and rf['all_finite']:
            difference=x.astype(np.float64)-y.astype(np.float64)
            row.update(max_abs=float(np.abs(difference).max()),relative_l2=float(np.linalg.norm(difference.ravel())/max(np.linalg.norm(y.astype(np.float64).ravel()),1e-30)),failed_elements=int(np.count_nonzero(np.abs(difference)>1e-5+1e-4*np.abs(y.astype(np.float64)))))
        else:row.update(max_abs=None,relative_l2=None,failed_elements=None)
        rows.append(row)
    bad=[row['name'] for row in rows[:len(ORDER)] if not row['SDK']['all_finite']]
    output=dict(schema='v29_saved_Update_tap_audit',status='passed_evidence_checks',auditor_sha256=sha(__file__),
        host_model_OR_ORT_execution=False,raw_arrays_modified=False,realtime_validated=False,
        manifest_sha256=ort['manifest_sha256'],receipt_sha256=dict(ORT=pins[str(a.ort_dir/'run.json')],SDK=pins[str(a.sdk_dir/'run.json')]),
        first_exposed_nonfinite_tensor=bad[0] if bad else None,
        interpretation='Only earliest among the declared exposed tensors in this instrumented graph; output taps can alter compiler fusion. This does not prove the first faulty internal kernel in the original graph or a unique root cause.',
        rtol=1e-4,atol=1e-5,rows=rows,SDK_candidate_accepted=sdk['candidate_accepted'],
        sources_and_saved_arrays_unchanged=all(sha(path)==digest for path,digest in pins.items()))
    assert output['sources_and_saved_arrays_unchanged'];a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'first_exposed_nonfinite_tensor':output['first_exposed_nonfinite_tensor'],'report_sha256':sha(a.output)}))
if __name__=='__main__':main()
