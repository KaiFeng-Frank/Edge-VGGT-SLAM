#!/usr/bin/env python3
"""Board-only SDK diagnostic of the fixed complete-source-frame Update.

No model adoption, BA or pose. Ordinary numerical/domain failures preserve raw
outputs; profile mode may observe one invalid call only under an explicit flag.
"""
import argparse,hashlib,importlib.util,json,os,platform,resource,sys,time,traceback
from pathlib import Path
import numpy as np

BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
PARENT_IO_SHA='a5af6d39b2d19471255fc86aa42c9aa9b0247e428715c23ffd4ba79b184ad865'
CONFIG=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False)
INPUTS=('net','inp','corr');OUTPUTS=('net_out','delta','weight')
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):value.update(chunk)
    return value.hexdigest()
def meta(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
def schedule(profile,invalid):
    if invalid:return [(0,'single_invalid_output_profile_diagnostic')]
    if profile:return [(0,'profile'),(1,'profile')]
    return [(case,role) for case in (0,1) for role in ('first','warm','warm')]
def statistics(value):
    finite=np.isfinite(value)
    return dict(elements=value.size,finite=int(finite.sum()),nan=int(np.isnan(value).sum()),
                positive_inf=int(np.isposinf(value).sum()),negative_inf=int(np.isneginf(value).sum()),
                all_zero=bool(np.all(value==0)),minimum=float(value.min()) if finite.all() else None,
                maximum=float(value.max()) if finite.all() else None)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('export-dir','compile-dir','io-helper','compiler-source','abi-helper','header','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('export','compile','io-helper','compiler-source'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--collect-perf-detail',action='store_true');p.add_argument('--diagnostic-observe-one-invalid-output',action='store_true')
    p.add_argument('--diagnostic-allow-fp32-rejected',action='store_true')
    a=p.parse_args(argv)
    require(platform.system()=='Linux' and platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'Actual SDK execution is RK3588-only')
    require(not a.diagnostic_observe_one_invalid_output or a.collect_perf_detail,'Invalid output observation is a single profile-only diagnostic')
    out=a.output_dir.resolve();require(out.is_relative_to(BASE) and out!=BASE,'Isolated output path required')
    protected=[v.resolve() for k,v in vars(a).items() if k!='output_dir' and isinstance(v,Path)]
    require(not any(out==v or out.is_relative_to(v) or v.is_relative_to(out) for v in protected),'Output overlaps source/input')
    out.mkdir(exist_ok=False);started=time.monotonic();affinity=os.sched_getaffinity(0);pins={};session=None
    report=dict(schema='v35_source_frame_channel_first_RKNN_board_diagnostic',experiment_version=35,status='failed',script_sha256=sha(__file__),
                actual_inference_on_RK3588S=True,host_model_inference=False,original_VGGT_default_unchanged=True,
                different_model_from_VGGT=True,ORB_used=False,added_LK_PnP_frontend=False,NPU_used=False,
                candidate_accepted=False,complete_VO_executed=False,BA_executed=False,realtime_validated=False,trajectory_accuracy_validated=False,
                collect_perf_detail=a.collect_perf_detail,diagnostic_observe_one_invalid_output=a.diagnostic_observe_one_invalid_output,
                calls=[],cleanup_errors=[],ordinary_schedule='Each of two fixed real states: first,warm,warm; separate profile process.',
                timing_scope='Complete synchronous SDK packing/set/run/get/owned output copy/release; post-call hashes/checks and full adapter wall time separately reported; no preprocessing, fullgraph, correlation, BA or pose.')
    def pin(path,expected):
        path=Path(path).resolve();require(sha(path)==expected,'Pinned source/input differs: '+str(path));pins[str(path)]=expected;return path
    def load_artifact(item):
        require(Path(item['file']).name==item['file'],'Artifact basename required');path=pin(a.export_dir/item['file'],item['sha256'])
        require(path.stat().st_size==item['bytes'],'Artifact size differs')
        with np.load(path,allow_pickle=False) as saved:values={k:saved[k].copy() for k in saved.files}
        require({k:meta(v) for k,v in values.items()}==item['arrays'],'Raw artifact arrays differ');return values
    def save(name,values):
        path=out/name
        with path.open('xb') as stream:np.savez(stream,**values)
        return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={k:meta(v) for k,v in values.items()})
    try:
        pin(__file__,report['script_sha256']);io=module(pin(a.io_helper,a.io_helper_sha256),'_v33_frozen_IO')
        pin(a.io_helper.parent/'v33_rknn_source_frame_padded.py',PARENT_IO_SHA)
        pin(a.compiler_source,a.compiler_source_sha256)
        require(io.PARENT_SHA256==PARENT_IO_SHA,'Runtime adapter ancestry differs')
        report.update(runtime_adapter_revision=3,runtime_adapter_parent_sha256=PARENT_IO_SHA,
                      logical_layout='BCDM_channel_first',diagnostic_allow_fp32_rejected=a.diagnostic_allow_fp32_rejected,
                      output_adapter_change='CF shape contract; unchanged native rank4 transport and actual logical 4*N returned bytes, dtype/shape, ownership and output-padding guards.')
        manifest,onnx_path,provenance=io.validate_manifest(a.export_dir/'manifest.json',a.export_sha256,verify_source_files=True,allow_fp32_rejected_diagnostic=a.diagnostic_allow_fp32_rejected)
        pins.update(provenance)
        compiled=json.loads(pin(a.compile_dir/'compile.json',a.compile_sha256).read_text())
        require(compiled['schema']=='v35_channel_first_Update_RKNN_compilation' and compiled['script_sha256']==a.compiler_source_sha256,'CF compiler identity differs')
        require(compiled['status']=='compiled_source_frame_diagnostic_not_runtime_validated' and compiled['released']
                and compiled['sources_and_inputs_unchanged'] and compiled['original_parameters_unchanged'] and not compiled['candidate_accepted'],'Completed matching diagnostic compilation required')
        require(compiled['configuration']==CONFIG and compiled['manifest_sha256']==a.export_sha256 and compiled['io_helper_sha256']==a.io_helper_sha256,'Compile configuration/provenance differs')
        require(compiled['input']['sha256']==manifest['model']['sha256'] and compiled['inputs']==manifest['inputs'] and compiled['outputs']==manifest['outputs'],'Compile model/IO differs')
        item=compiled['output'];require(Path(item['file']).name==item['file'],'RKNN basename required')
        model=pin(a.compile_dir/item['file'],item['sha256']);require(model.stat().st_size==item['bytes'],'RKNN size differs')
        cases=manifest['cases'];require(len(cases)==2,'Exactly two real reference cases required')
        feeds=[load_artifact(c['inputs_artifact']) for c in cases];refs=[load_artifact(c['reference_artifact']) for c in cases]
        ort_refs=[load_artifact(c['ort_artifact']) for c in cases]
        require({4,5,6,7}<=affinity,'Four A76 CPUs required');os.sched_setaffinity(0,{4,5,6,7})
        report.update(export_sha256=a.export_sha256,compile_sha256=a.compile_sha256,io_helper_sha256=a.io_helper_sha256,
                      source_plans=[c['source_plan'] for c in cases],inputs=manifest['inputs'],outputs=manifest['outputs'],
                      export_strict_FP32_gate_passed=manifest['strict_FP32_gate_passed'],configuration=CONFIG,cpu_affinity=sorted(os.sched_getaffinity(0)),numpy_version=np.__version__)
        # Large historical input files are verified before/after the experiment;
        # do not repeatedly read hundreds of MB between each SDK invocation.
        local_pins={str(Path(__file__).resolve()):report['script_sha256'],str(a.io_helper.resolve()):a.io_helper_sha256,
                    str((a.export_dir/'manifest.json').resolve()):a.export_sha256,str((a.compile_dir/'compile.json').resolve()):a.compile_sha256}
        session=io.SourceFrameSession.open_board(model,item['sha256'],manifest['inputs'],manifest['outputs'],abi_path=a.abi_helper,
                    header_path=a.header,library_path=a.library,pins=local_pins,collect_perf_detail=a.collect_perf_detail,log_directory=out)
        seen={};strict=[]
        for index,(case_index,role) in enumerate(schedule(a.collect_perf_detail,a.diagnostic_observe_one_invalid_output)):
            begin=time.monotonic();values,record=session.execute(feeds[case_index],refs[case_index]);elapsed=time.monotonic()-begin
            if set(values)==set(OUTPUTS):values={k:values[k] for k in OUTPUTS}
            record.update(index=index,case_index=case_index,timing_role=role,adapter_execute_wall_seconds=elapsed,
                          arrays={k:meta(v) for k,v in values.items()},statistics={k:statistics(v) for k,v in values.items()})
            report['calls'].append(record);report['NPU_used']=bool(session.metrics()['npu_used'])
            valid=(record.get('outputs_finite') and record.get('weight_domain_valid') and not record.get('all_outputs_zero'))
            if case_index not in seen or not valid:record['artifact']=save('call-%d-output.npz'%index,values)
            if case_index in seen:record['repeat_raw_hash_equal']=record['arrays']==seen[case_index]
            else:seen[case_index]=record['arrays']
            require(record['execution_status']=='completed','SDK execution failure: '+str(record.get('error')))
            require(tuple(values)==OUTPUTS,'Missing/reordered SDK outputs')
            record['versus_CPU_ORT']={k:io.compare(values[k],ort_refs[case_index][k]) for k in OUTPUTS}
            strict.append(record['strict_FP32_gate_passed'])
            require(valid or a.diagnostic_observe_one_invalid_output,'Invalid raw SDK outputs; saved before stopping')
        report.update(status='completed_channel_first_RKNN_diagnostic_not_adopted',actual_strict_FP32_gate_passed=all(strict),
                      all_outputs_finite=all(c['outputs_finite'] for c in report['calls']),
                      all_weight_domains_valid=all(c['weight_domain_valid'] for c in report['calls']))
        report['timing']={str(i):dict(first_seconds=next((c['timing']['forward_seconds'] for c in report['calls'] if c['case_index']==i and c['timing_role']=='first'),None),
                           warm_seconds=[c['timing']['forward_seconds'] for c in report['calls'] if c['case_index']==i and c['timing_role']=='warm']) for i in range(2)}
        for value in report['timing'].values():value['warm_mean_seconds']=sum(value['warm_seconds'])/len(value['warm_seconds']) if value['warm_seconds'] else None
    except Exception as error:report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if session is not None:
            session.close();report['session']=session.metrics();report['NPU_used']=bool(report['session']['npu_used'])
            if report['session']['cleanup_errors']:report['cleanup_errors'].extend(report['session']['cleanup_errors']);report['status']='failed'
        try:
            report['source_hashes_after']={p:sha(p) for p in pins};report['sources_and_inputs_unchanged']=report['source_hashes_after']==pins;require(report['sources_and_inputs_unchanged'],'Source/input changed')
        except Exception as error:report['cleanup_errors'].append(repr(error));report['status']='failed'
        try:os.sched_setaffinity(0,affinity);report['affinity_restored']=set(os.sched_getaffinity(0))==set(affinity);require(report['affinity_restored'],'Affinity restore failed')
        except Exception as error:report['cleanup_errors'].append(repr(error));report['status']='failed'
        report.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,actual_driver_seconds=time.monotonic()-started)
        (out/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','actual_strict_FP32_gate_passed','all_outputs_finite','all_weight_domains_valid','timing','cleanup_errors')}),flush=True)
    return 0 if report['status']=='completed_channel_first_RKNN_diagnostic_not_adopted' else 1
if __name__=='__main__':raise SystemExit(main())
