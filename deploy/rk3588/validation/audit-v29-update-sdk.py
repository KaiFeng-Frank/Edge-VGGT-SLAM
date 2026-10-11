#!/usr/bin/env python3
"""NumPy-only saved v29 SDK evidence audit; no SDK, ORT or model execution."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

sys.dont_write_bytecode=True
PARENT_SHA='9782e40a91ec1f0995599c09acd0f960ef44445eaef272dd395cc39ab5d0f701'
COMPILER_SHA='a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
HEADER_SHA='c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIB_SHA='d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
COLUMNS=('ID','OpType','DataType','Target','InputShape','OutputShape','Cycles(DDR/NPU/Total)',
         'Time(us)','MacUsage(%)','WorkLoad(0/1/2)','RW(KB)','FullName')
STAGES=('pack_seconds','input_set_seconds','run_api_seconds','get_seconds','copy_seconds','release_seconds')


def load_parent(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=PARENT_SHA:raise ValueError('Frozen NumPy parent SHA differs')
    spec=importlib.util.spec_from_file_location('_v29_saved_ort_audit',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def parse_profile(profile):
    raw=profile['raw_text'];blob=raw.encode('utf-8')
    if profile['query_command']!=3 or profile['bytes']!=len(blob) or profile['raw_sha256']!=hashlib.sha256(blob).hexdigest():
        raise ValueError('Profile raw byte/hash/query binding differs')
    lines=raw.splitlines();headers=[i for i,x in enumerate(lines) if x.startswith('ID   OpType')]
    if len(headers)!=1:raise ValueError('Unique layer table header required')
    header=lines[headers[0]]
    if tuple(header.split())!=COLUMNS:raise ValueError('Unexpected SDK layer table columns')
    offsets=[header.index(x) for x in COLUMNS];rows=[]
    start=headers[0]+2
    for line in lines[start:]:
        if re.fullmatch('-+',line):break
        values={key:line[at:offsets[i+1] if i+1<len(offsets) else None].strip() for i,(key,at) in enumerate(zip(COLUMNS,offsets))}
        if not re.fullmatch(r'\d+',values['ID']) or not re.fullmatch(r'\d+',values['Time(us)']) or values['Target'] not in ('CPU','GPU','NPU'):
            raise ValueError('Malformed SDK operator row')
        if int(values['ID'])!=len(rows)+1 or not values['OpType'] or not values['FullName']:raise ValueError('Missing/duplicate SDK operator row')
        rows.append(dict(index=int(values['ID']),op=values['OpType'],target=values['Target'],time_us=int(values['Time(us)']),name=values['FullName']))
    if not rows:raise ValueError('Empty SDK layer table')
    elapsed=re.findall(r'^Total Operator Elapsed Per Frame Time\(us\): (\d+)$',raw,re.M)
    targets={key:sum(x['time_us'] for x in rows if x['target']==key) for key in ('CPU','GPU','NPU')}
    if elapsed!=[str(sum(targets.values()))]:raise ValueError('Layer time total differs')
    ranking='OpType CallNumber CPUTime(us) GPUTime(us) NPUTime(us) TotalTime(us) TimeRatio(%)'
    ranks=[i for i,x in enumerate(lines) if ' '.join(x.split())==ranking]
    if len(ranks)!=1:raise ValueError('Unique operator ranking header required')
    actual_ops={};declared_total=None
    for line in lines[ranks[0]+2:]:
        if re.fullmatch('-+',line):
            if declared_total is not None:break
            continue
        total=re.fullmatch(r'Total\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*',line)
        if total:
            declared_total=[int(x) for x in total.groups()];continue
        item=re.fullmatch(r'(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)%\s*',line)
        if not item:raise ValueError('Malformed SDK operator ranking row')
        op,count,cpu,gpu,npu,whole,ratio=item.groups()
        if op in actual_ops:raise ValueError('Duplicate ranked operator')
        matching=[x for x in rows if x['op']==op]
        calculated=[len(matching)]+[sum(x['time_us'] for x in matching if x['target']==t) for t in ('CPU','GPU','NPU')]
        if calculated!=[int(count),int(cpu),int(gpu),int(npu)] or int(whole)!=sum(calculated[1:]):raise ValueError('Ranking differs from layer rows')
        actual_ops[op]=dict(count=int(count),CPU_us=int(cpu),GPU_us=int(gpu),NPU_us=int(npu),total_us=int(whole))
    if set(actual_ops)!={x['op'] for x in rows} or declared_total!=[targets['CPU'],targets['GPU'],targets['NPU'],sum(targets.values())]:
        raise ValueError('Ranking total/coverage differs')
    internal=[x for x in rows if x['op'] not in ('InputOperator','OutputOperator')]
    return dict(raw_sha256=profile['raw_sha256'],row_count=len(rows),target_counts=dict(collections.Counter(x['target'] for x in rows)),
        internal_target_counts=dict(collections.Counter(x['target'] for x in internal)),target_time_us=targets,total_operator_us=sum(targets.values()),
        operator_summary=actual_ops,hotspots=sorted(rows,key=lambda x:x['time_us'],reverse=True)[:10],
        source='Actual SDK perf-detail rows, independently reconciled to both SDK totals; no compiler placement inference.')


def weight_domain(weight):
    return bool(np.isfinite(weight).all() and ((weight>=0)&(weight<=1)).all())


def input_records_valid(contracts,records,attrs):
    if len(contracts)!=13 or len(records)!=13 or len(attrs)!=13:return False
    for i,(logical,record,attr) in enumerate(zip(contracts,records,attrs)):
        isint=logical['dtype']==7;transport='int64' if isint else 'float32';size=math.prod(logical['shape'])
        if not (record['index']==i and record['name']==logical['name'] and record['logical_shape']==logical['shape']
            and record['source_dtype']==transport and record['transport_dtype']==transport and record['transport_type']==(8 if isint else 0)
            and record['bytes']==size*(8 if isint else 4) and record['format']==3 and record['pass_through']==0 and record['owned']
            and record['exact_integer_cast_verified']==isint):return False
        if not (attr['index']==i and attr['name']==logical['name'] and attr['shape']==logical['shape'] and attr['n_elems']==size
            and attr['format']==3 and attr['type']==(8 if isint else 1) and attr['size']==size*(8 if isint else 2)
            and attr['quantization_type']==2 and attr['scale']==1 and attr['zero_point']==0 and attr['fractional_length']==0):return False
    return True


def audit_candidate(label,run_dir,manifest_dir,ort_dir,capture_dir,export_dir,audit,parent,capture,export):
    run_dir,manifest_dir,ort_dir=map(Path,(run_dir,manifest_dir,ort_dir))
    report=json.loads((run_dir/'run.json').read_text());manifest=json.loads((manifest_dir/'manifest.json').read_text());ort=json.loads((ort_dir/'run.json').read_text())
    identity=dict(run_sha256=audit.file(run_dir/'run.json'),manifest_sha256=audit.file(manifest_dir/'manifest.json',report['manifest_sha256']),ORT_run_sha256=audit.file(ort_dir/'run.json'))
    audit.check(report['schema']=='v29_Update_candidate_board_diagnostic' and report['mode'] in ('sdk-normal','sdk-profile'),'SDK_schema:'+label)
    audit.check(report['script_sha256']==parent.RUNNER_SHA and report['capture_sha256']==parent.CAPTURE_SHA and report['export_sha256']==parent.EXPORT_SHA,'runner_parents:'+label)
    audit.check(report['execution_status']=='completed' and report['cleanup_errors']==[] and report['destroy_code']==0 and report['affinity_restored'],'execution_cleanup:'+label)
    audit.check(report['source_pins']==report['source_hashes_after'] and report['sources_and_inputs_unchanged'],'recorded_source_after:'+label)
    known=dict(parent.HELPERS,**{'run-v29-update-candidate-board.py':parent.RUNNER_SHA,'compile-v12-rknn-block.py':COMPILER_SHA,'rknn_api_v2.3.2.h':HEADER_SHA,'librknnrt.so':LIB_SHA})
    for name,value in known.items():audit.check([v for p,v in report['source_pins'].items() if Path(p).name==name]==[value],'recorded_helper_SHA:'+label+':'+name)
    audit.check(report['actual_board_execution'] and not any(report[k] for k in ('CPU_model_fallback','production_changed','BA_executed','ORB_used','valid_pose_output','realtime_validated','FP32_thresholds_relaxed','outputs_clipped_or_modified')),'honest_scope:'+label)
    audit.check(report['versions']['api'].startswith('2.3.2') and report['core_mask']==7 and report['init_flags']==(8 if report['mode']=='sdk-profile' else 0),'SDK_mode:'+label)
    audit.check(manifest['schema']=='v29_DPVO_Update_candidate_manifest' and manifest['source_capture_sha256']==parent.CAPTURE_SHA and manifest['source_export_sha256']==parent.EXPORT_SHA,'manifest_parents:'+label)
    audit.check(ort['mode']=='ort' and ort['execution_status']=='completed' and ort['manifest_sha256']==identity['manifest_sha256'] and ort['source_pins']==ort['source_hashes_after'] and not ort['cleanup_errors'],'same_candidate_ORT:'+label)
    audit.check(report['ORT_candidate_accepted']==ort['candidate_accepted'] and (ort['candidate_accepted'] or report['explicit_unvalidated_ORT_diagnostic']),'explicit_unvalidated_diagnostic:'+label)
    keys=report['model_keys'];audit.check(len(keys)==1,'single_context_model:'+label);key=keys[0];model=manifest['models'][key];original_model=export['models'][key]
    audit.check(model['inputs']==original_model['inputs'] and model['outputs']==original_model['outputs'] and model['source_onnx_sha256']==original_model['sha256'],'unchanged_IO_parent:'+label)
    for folder,row in ((manifest_dir,model),(Path(export_dir),original_model)):
        audit.file(folder/row['file'],row['sha256']);audit.check((folder/row['file']).stat().st_size==row['bytes'],'graph_bytes:'+label)
    compiled=[(p,v) for p,v in report['source_pins'].items() if Path(p).name=='compile.json']
    if len(compiled)!=1:raise ValueError('Unique compile receipt pin required')
    compile_dir=run_dir.parent/Path(compiled[0][0]).parent.name;compile_path=compile_dir/'compile.json';audit.file(compile_path,compiled[0][1]);compiler=json.loads(compile_path.read_text())
    audit.check(compiler['status']=='compiled_not_runtime_validated' and compiler['script_sha256']==COMPILER_SHA and compiler['input']['sha256']==model['sha256'] and compiler['released'],'compile_chain:'+label)
    audit.check(all(compiler['configuration'].get(k)==v for k,v in dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False).items()),'original_compile_config:'+label)
    binary=compile_dir/Path(compiler['output']['path']).name;audit.file(binary,compiler['output']['sha256']);audit.check(binary.stat().st_size==compiler['output']['size'],'compiled_bytes:'+label)
    audit.check([v for p,v in report['source_pins'].items() if Path(p).name==binary.name]==[compiler['output']['sha256']],'compiled_pin:'+label)
    expected=parent.expected_states(capture,export);selected=[x for x in expected if x[0]==key]
    schedule=selected[:1] if report['mode']=='sdk-profile' or report['first_only'] else [selected[0]]*4+selected[1:]
    audit.check(report['selected_saved_states']==len(selected) and [(x['model_key'],x['case'],x['step']) for x in report['calls']]==schedule,'complete_declared_schedule:'+label)
    ort_states={(x['model_key'],x['case'],x['step']):x for x in ort['calls']}
    audit.check(len(ort_states)==len(ort['calls']) and set(schedule)<=set(ort_states),'ORT_reference_states:'+label)
    native=report['native_log'];path=run_dir/native['file'];audit.file(path,native['sha256']);text=path.read_text(errors='replace');text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
    errors=[line for line in text.splitlines() if re.search(r'\bE\s+RKNN\b',line)]
    audit.check(path.stat().st_size==native['bytes'] and errors==native['errors']==[],'no_hidden_native_errors:'+label)
    rows=[];seen={};all_accepted=bool(ort['candidate_accepted'])
    for i,call in enumerate(report['calls']):
        statekey=(call['model_key'],call['case'],call['step']);original,wrapped,topology=expected[statekey]
        audit.check(call['index']==i,'call_index:'+label+':'+str(i))
        audit.artifact(capture_dir,original['inputs']);audit.artifact(export_dir,topology)
        ref=audit.artifact(capture_dir,original['outputs']);wrap=audit.artifact(export_dir,wrapped['outputs']);candidate=audit.artifact(ort_dir,ort_states[statekey]['outputs']);actual=audit.artifact(run_dir,call['outputs'])
        audit.check(tuple(actual)==parent.CORE and all(v.dtype==np.float32 and list(v.shape)==row['shape'] for v,row in zip(actual.values(),model['outputs'])),'FP32_output_contract:'+label+':'+str(i))
        counters={k:parent.finite(v) for k,v in actual.items()};finite=all(x['all_finite'] for x in counters.values());domain=weight_domain(actual['weight'])
        checks={k:parent.residual(actual[n],ref[k]) for n,k in zip(parent.CORE,parent.REF)};wrapped_checks={k:parent.residual(actual[n],wrap[k]) for n,k in zip(parent.CORE,parent.REF)};ort_checks={k:parent.residual(v,candidate[k]) for k,v in actual.items()}
        strict=all(x['gate_passed'] for x in (*checks.values(),*wrapped_checks.values()));ort_strict=all(x['gate_passed'] for x in ort_checks.values())
        audit.check(counters==call['finite_counts'] and finite==call['all_outputs_finite'] and domain==call['weight_domain_passed'],'independent_finite_domain:'+label+':'+str(i))
        audit.check(all(parent.recorded_residual_matches(checks[k],call['versus_original'][k]) and parent.recorded_residual_matches(wrapped_checks[k],call['versus_wrapper'][k]) for k in parent.REF)
            and all(parent.recorded_residual_matches(ort_checks[k],call['versus_candidate_ORT'][k]) for k in parent.CORE)
            and strict==call['core_strict_FP32_gate_passed'] and ort_strict==call['candidate_ORT_strict_FP32_gate_passed'],'independent_strict_residuals:'+label+':'+str(i))
        signature={k:parent.metadata(v) for k,v in actual.items()};repeat=statekey not in seen or seen[statekey]==signature;seen[statekey]=signature
        audit.check(repeat==call['repeat_bitwise_equal'],'repeat_raw_bytes:'+label+':'+str(i))
        timing=call['timing'];audit.check(all(math.isfinite(timing[k]) and timing[k]>=0 for k in (*STAGES,'forward_seconds')) and timing['forward_seconds']+1e-9>=sum(timing[k] for k in STAGES) and timing['outputs_released'],'synchronous_owned_copy_release_timing:'+label+':'+str(i))
        audit.check(input_records_valid(model['inputs'],timing['input_records'],report['input_attributes']),'mixed_transport_contract:'+label+':'+str(i))
        audit.check(all(a['index']==j and a['name']==c['name'] and a['shape']==c['shape'] and a['format']==3 and a['type']==1 and a['n_elems']==math.prod(c['shape']) and a['size']==2*math.prod(c['shape']) and a['quantization_type']==2 and a['scale']==1 and a['zero_point']==0 and a['fractional_length']==0 for j,(a,c) in enumerate(zip(report['output_attributes'],model['outputs']))) and len(report['output_attributes'])==3,'native_output_attributes:'+label)
        all_accepted &= finite and domain and strict and ort_strict and repeat
        entry=dict(index=i,case=call['case'],step=call['step'],outputs_sha256=call['outputs']['sha256'],all_outputs_finite=finite,finite_counts=counters,weight_domain_passed=domain,
                   core_strict_FP32_gate_passed=strict,candidate_ORT_strict_FP32_gate_passed=ort_strict,repeat_bitwise_equal=repeat,versus_original=checks,versus_wrapper=wrapped_checks,versus_candidate_ORT=ort_checks)
        if report['mode']=='sdk-profile':entry['profile']=parse_profile(call['profile']);audit.check(call['profile']['operator_targets_interpreted_by_probe'] is False,'actual_profile_reinterpreted_by_auditor:'+label)
        else:audit.check('profile' not in call,'normal_not_profile:'+label)
        rows.append(entry)
    audit.check(report['candidate_accepted']==all_accepted and report['numerical_status']==('accepted' if all_accepted else 'rejected') and report['status']==('completed_numerical_accepted' if all_accepted else 'completed_numerical_rejected'),'reported_acceptance:'+label)
    times=[x['timing']['forward_seconds'] for x in report['calls']];warm=times[1:4] if len(times)>=4 else []
    return dict(label=label,identity=identity,mode=report['mode'],SDK_calls=len(rows),saved_arrays=len(rows)*3,finite_call_count=sum(x['all_outputs_finite'] for x in rows),domain_passed_call_count=sum(x['weight_domain_passed'] for x in rows),strict_FP32_passed_call_count=sum(x['core_strict_FP32_gate_passed'] for x in rows),candidate_accepted=bool(all_accepted),memory=report['memory'],cleanup_errors=report['cleanup_errors'],destroy_code=report['destroy_code'],
        timing=dict(all_forward_seconds=times,first_seconds=times[0],first_state_three_warm_mean_seconds=float(np.mean(warm)) if warm else None,timing_scope=report['timing_scope']),calls=rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture-dir',type=Path,required=True);p.add_argument('--export-dir',type=Path,required=True)
    p.add_argument('--candidate',nargs=4,action='append',metavar=('LABEL','SDK_RUN','MANIFEST_DIR','ORT_RUN'),required=True)
    p.add_argument('--parent-auditor',type=Path,default=Path(__file__).with_name('audit-v29-update-ort.py'));p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    report=dict(schema='v29_saved_Update_SDK_independent_review',status='failed',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),NumPy_only=True,host_ORT_execution=False,host_model_forward=False,board_execution=False,candidates=[]);audit=None
    try:
        parent=load_parent(args.parent_auditor);audit=parent.Audit();audit.file(args.parent_auditor,PARENT_SHA)
        audit.file(args.capture_dir/'capture.json',parent.CAPTURE_SHA);audit.file(args.export_dir/'export.json',parent.EXPORT_SHA)
        capture=json.loads((args.capture_dir/'capture.json').read_text());export=json.loads((args.export_dir/'export.json').read_text())
        for row in args.candidate:report['candidates'].append(audit_candidate(*row,args.capture_dir,args.export_dir,audit,parent,capture,export))
        report['cross_process_same_state_raw_equal']=[]
        for i,one in enumerate(report['candidates']):
            for two in report['candidates'][i+1:]:
                if one['identity']['manifest_sha256']!=two['identity']['manifest_sha256']:continue
                common={(x['case'],x['step']) for x in one['calls']} & {(x['case'],x['step']) for x in two['calls']}
                for key in sorted(common):
                    a=next(x for x in one['calls'] if (x['case'],x['step'])==key);b=next(x for x in two['calls'] if (x['case'],x['step'])==key)
                    paths=[Path(row[1])/f'call-{item["index"]:02d}.npz' for row,item in ((args.candidate[i],a),(args.candidate[report['candidates'].index(two)],b))]
                    with np.load(paths[0],allow_pickle=False) as aa,np.load(paths[1],allow_pickle=False) as bb:equal={k:aa[k].tobytes()==bb[k].tobytes() for k in aa.files}
                    report['cross_process_same_state_raw_equal'].append(dict(labels=[one['label'],two['label']],case=key[0],step=key[1],array_equal=equal))
        after={path:parent.digest(path) for path in audit.files};audit.check(after==audit.files,'all_actual_audited_files_unchanged');report['inputs_unchanged']=after==audit.files
        report['input_hashes']={Path(path).name+':'+str(i):value for i,(path,value) in enumerate(audit.files.items())}
        report['status']='passed' if not audit.failures else 'failed'
    except Exception as error:report['error']=repr(error)
    report['checks']=dict(count=audit.count if audit else 0,failed=len(audit.failures) if audit else 1,failures=audit.failures if audit else ['parent_initialization'])
    report['limits']=['Passed verifies saved evidence consistency; strict/domain/nonfinite rejection is retained and no candidate is promoted.',
        'Strict FP32 rtol=1e-4/atol=1e-5 and exact finite [0,1] weight gates remain unchanged. Derived L2 scalar reports allow only the frozen parent norm-rounding tolerance.',
        'Each SDK call evaluates a previously saved input independently, not recursively advancing a trajectory or BA.',
        'Profile rows describe actual SDK targets, including internal CPU work; operator sums are diagnostic and are not normal end-to-end accepted-pose latency.',
        'Normal timings include owned packing, set/run/get/copy/release; initialization, reference loading, hashing/statistics and disk I/O are excluded. Separate processes are observations, not a controlled benchmark.',
        'Source/header/library before-after pins are recorded board evidence; actual saved artifacts and graph/model/compile files are rehashed on this host without loading SDK.',
        'No full VO, accuracy acceptance or 20 Hz validation is claimed.']
    with args.output.open('x') as file:json.dump(report,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report['checks'],error=report.get('error'),output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()),allow_nan=False))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
