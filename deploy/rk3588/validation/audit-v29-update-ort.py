#!/usr/bin/env python3
"""Independent saved-array audit. NumPy only; never runs ORT/Torch/SDK."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode=True
CAPTURE_SHA='87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
EXPORT_SHA='56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac'
RUNNER_SHA='fe2e8b304ee1ef5c12b776630b913be401d29d6dde656e8c05211f7144704a92'
HELPERS={'v28_rknn_inputs.py':'040f55b89accb7a03a523ef2ab8c4bf12755f15da34813dc6ee607b6deb90c88',
         'run-v12-rknn-block-board.py':'37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'}
CORE=('net_out','delta','weight');REF=('net','delta','weight')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def metadata(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,
                tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def residual(a,b):
    if a.dtype!=np.float32 or b.dtype!=np.float32 or a.shape!=b.shape:raise ValueError('FP32 shape/dtype differs')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return dict(rtol=1e-4,atol=1e-5,gate_passed=False,nonfinite_pairs=int((~(np.isfinite(a)&np.isfinite(b))).sum()),
                    failed_elements=None,max_abs=None,relative_l2=None,comparison_undefined=True)
    av,bv=a.astype(np.float64),b.astype(np.float64);d=av-bv
    failed=int(np.count_nonzero(~np.isclose(av,bv,rtol=1e-4,atol=1e-5)))
    return dict(rtol=1e-4,atol=1e-5,gate_passed=failed==0,failed_elements=failed,
                max_abs=float(abs(d).max()),relative_l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(bv.ravel()),1e-30)),
                bitwise_equal=a.tobytes()==b.tobytes())


def finite(a):
    good=np.isfinite(a);v=a[good]
    return dict(elements=a.size,finite=int(good.sum()),nan=int(np.isnan(a).sum()),
        positive_inf=int(np.isposinf(a).sum()),negative_inf=int(np.isneginf(a).sum()),
        all_finite=bool(good.all()),all_zero=bool((a==0).all()),
        finite_min=float(v.min()) if v.size else None,finite_max=float(v.max()) if v.size else None)


def recorded_residual_matches(actual,recorded):
    # ARM/host BLAS norm reduction has tiny rounding differences. This applies
    # only to an already derived scalar report; all strict element decisions,
    # failed counts, max_abs and bitwise flags remain exact and unrelaxed.
    if set(actual)!=set(recorded):return False
    for key,value in actual.items():
        if key=='relative_l2' and value is not None and recorded[key] is not None:
            if not math.isclose(value,recorded[key],rel_tol=2e-12,abs_tol=1e-20):return False
        elif value!=recorded[key]:return False
    return True


class Audit:
    def __init__(self):self.count=0;self.failures=[];self.files={}
    def check(self,ok,name):
        self.count+=1
        if not ok:self.failures.append(name)
    def file(self,path,expected=None):
        path=Path(path);actual=digest(path)
        if expected is not None:self.check(actual==expected,'file_SHA:'+path.name)
        self.files[str(path.resolve())]=actual
        return actual
    def artifact(self,folder,row):
        if Path(row['file']).name!=row['file']:raise ValueError('Artifact basename required')
        path=Path(folder)/row['file'];self.file(path,row['sha256'])
        self.check(path.stat().st_size==row['bytes'],'bytes:'+path.name)
        with np.load(path,allow_pickle=False) as archive:result={k:archive[k].copy() for k in archive.files}
        self.check({k:metadata(v) for k,v in result.items()}==row['arrays'],'raw_metadata:'+path.name)
        return result


def expected_states(capture,export):
    result={}
    for case in capture['scenarios']:
        other=next(x for x in export['scenarios'] if x['name']==case['name'])
        if [x['step'] for x in case['states']]!=list(range(4)):raise ValueError('Original case must contain four steps')
        for row in case['states']:
            wrapped=next(x for x in other['states'] if x['step']==row['step'])
            key=(other['model_key'],case['name'],row['step'])
            if key in result:raise ValueError('Duplicate original state')
            result[key]=(row,wrapped,other['topology'])
    if len(result)!=16:raise ValueError('Original selection must retain all sixteen states')
    return result


def timing_summary(calls):
    result={}
    for key in sorted({x['model_key'] for x in calls}):
        values=[x['timing']['forward_seconds'] for x in calls if x['model_key']==key]
        result[key]=dict(count=len(values),mean_seconds=float(np.mean(values)),minimum_seconds=min(values),maximum_seconds=max(values))
    return result


def audit_candidate(label,run_dir,manifest_dir,capture_dir,export_dir,audit,capture,export):
    run_dir,manifest_dir=Path(run_dir),Path(manifest_dir)
    report=json.loads((run_dir/'run.json').read_text());manifest=json.loads((manifest_dir/'manifest.json').read_text())
    identity=dict(run_sha256=audit.file(run_dir/'run.json'),manifest_sha256=audit.file(manifest_dir/'manifest.json',report['manifest_sha256']))
    audit.check(report['schema']=='v29_Update_candidate_board_diagnostic' and report['mode']=='ort','ORT_schema:'+label)
    audit.check(report['script_sha256']==RUNNER_SHA and report['capture_sha256']==CAPTURE_SHA and report['export_sha256']==EXPORT_SHA,'runner_parents:'+label)
    audit.check(report['execution_status']=='completed' and report['cleanup_errors']==[] and report['affinity_restored'],'execution_cleanup:'+label)
    audit.check(report['source_pins']==report['source_hashes_after'] and report['sources_and_inputs_unchanged'],'recorded_source_after:'+label)
    for name,value in dict(HELPERS,**{'run-v29-update-candidate-board.py':RUNNER_SHA}).items():
        values=[v for p,v in report['source_pins'].items() if Path(p).name==name]
        audit.check(values==[value],'recorded_helper_SHA:'+label+':'+name)
    audit.check(report['providers']==['CPUExecutionProvider'] and report['ORT_optimization']=='disabled','CPU_ORT_scope:'+label)
    audit.check(report['actual_board_execution'] and not any(report[k] for k in ('CPU_model_fallback','production_changed','BA_executed','ORB_used','valid_pose_output','realtime_validated','FP32_thresholds_relaxed','outputs_clipped_or_modified')),'honest_scope:'+label)
    audit.check(manifest['schema']=='v29_DPVO_Update_candidate_manifest' and manifest['source_capture_sha256']==CAPTURE_SHA
                and manifest['source_export_sha256']==EXPORT_SHA,'manifest_parents:'+label)
    expected=expected_states(capture,export)
    audit.check(set(manifest['models'])=={'E96','E864'} and not report['first_only'] and report['selected_saved_states']==16,'whole_state_selection:'+label)
    keys=[(r['model_key'],r['case'],r['step']) for r in report['calls']]
    audit.check(len(keys)==16 and set(keys)==set(expected) and len(set(keys))==16,'no_missing_duplicate_states:'+label)
    for key,model in manifest['models'].items():
        parent=export['models'][key]
        audit.check(model['source_onnx_sha256']==parent['sha256'] and model['inputs']==parent['inputs'] and model['outputs']==parent['outputs'],'original_IO:'+label+':'+key)
        for folder,row in ((manifest_dir,model),(Path(export_dir),parent)):
            if Path(row['file']).name!=row['file']:raise ValueError('Graph basename required')
            audit.file(folder/row['file'],row['sha256']);audit.check((folder/row['file']).stat().st_size==row['bytes'],'graph_bytes:'+label+':'+key)
    details=[];domain_bad=[];ranges=[];all_accepted=True
    for index,row in enumerate(report['calls']):
        key=(row['model_key'],row['case'],row['step']);original,wrapped,topology=expected[key]
        audit.check(row['index']==index,'call_index:'+label+':'+str(index))
        audit.artifact(capture_dir,original['inputs']);audit.artifact(export_dir,topology)
        ref=audit.artifact(capture_dir,original['outputs']);wrap=audit.artifact(export_dir,wrapped['outputs']);actual=audit.artifact(run_dir,row['outputs'])
        audit.check(tuple(actual)==CORE,'original_output_order:'+label+':'+str(index))
        counters={name:finite(value) for name,value in actual.items()};isfinite=all(x['all_finite'] for x in counters.values())
        checks={refname:residual(actual[name],ref[refname]) for name,refname in zip(CORE,REF)}
        wrapper_checks={refname:residual(actual[name],wrap[refname]) for name,refname in zip(CORE,REF)}
        strict=all(x['gate_passed'] for x in (*checks.values(),*wrapper_checks.values()))
        weight=actual['weight'];domain=bool(np.isfinite(weight).all() and ((weight>=0)&(weight<=1)).all())
        audit.check(counters==row['finite_counts'] and isfinite==row['all_outputs_finite'],'finite_counts:'+label+':'+str(index))
        audit.check(all(recorded_residual_matches(checks[name],row['versus_original'][name]) and recorded_residual_matches(wrapper_checks[name],row['versus_wrapper'][name]) for name in REF)
                    and strict==row['core_strict_FP32_gate_passed'],'independent_FP32_residuals:'+label+':'+str(index))
        audit.check(domain==row['weight_domain_passed'],'independent_weight_domain:'+label+':'+str(index))
        t=row['timing'];audit.check(all(math.isfinite(t[k]) and t[k]>=0 for k in ('pack_seconds','run_seconds','copy_seconds','forward_seconds'))
            and math.isclose(t['forward_seconds'],t['pack_seconds']+t['run_seconds']+t['copy_seconds'],rel_tol=1e-10,abs_tol=1e-10),'timing_arithmetic:'+label+':'+str(index))
        if not domain:domain_bad.append(dict(model_key=key[0],case=key[1],step=key[2],minimum=counters['weight']['finite_min'],maximum=counters['weight']['finite_max'],out_of_range_elements=int(((weight<0)|(weight>1)|~np.isfinite(weight)).sum())))
        ranges.extend([counters['weight']['finite_min'],counters['weight']['finite_max']]);all_accepted &= isfinite and strict and domain
        details.append(dict(model_key=key[0],case=key[1],step=key[2],core_strict_FP32_gate_passed=strict,weight_domain_passed=domain,
                            versus_original=checks,versus_wrapper=wrapper_checks,outputs_sha256=row['outputs']['sha256']))
    audit.check(report['candidate_accepted']==all_accepted and report['numerical_status']==('accepted' if all_accepted else 'rejected')
                and report['status']==('completed_numerical_accepted' if all_accepted else 'completed_numerical_rejected'),'reported_acceptance:'+label)
    return dict(label=label,identity=identity,state_count=len(keys),saved_output_arrays=len(keys)*3,
        strict_FP32_state_count=sum(x['core_strict_FP32_gate_passed'] for x in details),domain_passed_state_count=len(keys)-len(domain_bad),
        candidate_accepted=bool(all_accepted),weight_range=dict(minimum=min(ranges),maximum=max(ranges)),domain_failures=domain_bad,
        timing=timing_summary(report['calls']),timing_scope=report['timing_scope'],states=details)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture-dir',type=Path,required=True);p.add_argument('--export-dir',type=Path,required=True)
    p.add_argument('--candidate',nargs=3,action='append',metavar=('LABEL','RUN_DIR','MANIFEST_DIR'),required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);audit=Audit();report=dict(schema='v29_saved_Update_ORT_independent_review',status='failed',script_sha256=digest(__file__),
        NumPy_only=True,host_ORT_execution=False,host_model_forward=False,board_execution=False,candidates=[])
    try:
        audit.file(args.capture_dir/'capture.json',CAPTURE_SHA);audit.file(args.export_dir/'export.json',EXPORT_SHA)
        capture=json.loads((args.capture_dir/'capture.json').read_text());export=json.loads((args.export_dir/'export.json').read_text())
        for label,run,manifest in args.candidate:report['candidates'].append(audit_candidate(label,run,manifest,args.capture_dir,args.export_dir,audit,capture,export))
        after={path:digest(path) for path in audit.files};audit.check(after==audit.files,'all_actual_audited_files_unchanged')
        report['input_hashes']={str(Path(path).name)+':'+str(i):value for i,(path,value) in enumerate(audit.files.items())}
        report['inputs_unchanged']=after==audit.files
        report['status']='passed' if not audit.failures else 'failed'
    except Exception as error:report['error']=repr(error)
    report['checks']=dict(count=audit.count,failed=len(audit.failures),failures=audit.failures)
    report['limits']=['Passed means saved evidence and declared numerical decisions are consistent; a rejected candidate remains rejected.',
        'Strict FP32 gate is rtol=1e-4/atol=1e-5; exact finite [0,1] weight is an additional unrelaxed domain gate.',
        'Only comparison of recorded relative-L2 report scalars permits cross-platform norm rounding (rtol=2e-12/atol=1e-20); gates/counts/maxabs remain exact.',
        'All sixteen original saved states are independent comparisons, not recursive candidate trajectories.',
        'These are CPU ORT saved outputs, not SDK/NPU execution, BA feedback, VO, full-model quality or 20 Hz proof.',
        'Timing is board CPU ORT pack/run/owned-copy only. Topology/input-domain prevalidation, initialization, hashing/statistics/artifact I/O are excluded.',
        'Historical board source paths are compared through their recorded before/after pins. Actual host data/reference/graph files are independently rehashed.']
    with args.output.open('x') as file:json.dump(report,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report['checks'],output_sha256=digest(args.output)),allow_nan=False))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
