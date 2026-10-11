"""Read-only saved-array audit. NumPy/JSON only; no GT, forward or SDK."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CPU_SHA='33fb611edf250ae250f06c804507d2a7a180855c401e5457ab8a5ebf217b7f36'
BOTH_SHA='78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b'
ORIGINAL_SHA='dd9f37524b999c1e586182a92c226481fcfc7f4da851d5e88d62d82be1029ecd'
NORMAL_SHA='b170f07f94e95125664359e816777867c456f9777ca1fb3892900eb3fd614690'
EXPORT_SHA='5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
INPUTS={'net','inp','corr','ii','jj','kk'}
OUTPUTS={'net','delta','weight'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
        tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def equal(left,right):
    return left.shape==right.shape and left.dtype==right.dtype and left.tobytes()==right.tobytes()


def compare(actual,reference):
    assert actual.shape==reference.shape and actual.dtype==reference.dtype==np.float32
    a,b=actual.astype(np.float64),reference.astype(np.float64);difference=a-b
    failed=int(np.count_nonzero(np.abs(difference)>1e-5+1e-4*np.abs(b)))
    norm=float(np.linalg.norm(b.ravel()));error=float(np.linalg.norm(difference.ravel()))
    return dict(rtol=1e-4,atol=1e-5,gate_passed=failed==0,failed_elements=failed,
        element_count=actual.size,bitwise_equal=equal(actual,reference),
        max_abs=float(np.abs(difference).max()),rmse=float(np.sqrt(np.mean(difference*difference))),
        relative_l2=error/norm if norm else None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cpu-dir','both-dir','original-dir','v40-normal-dir','v40-export-manifest','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args(argv);pins={};artifacts=[]
    def pin(path,expected):
        path=Path(path).resolve();assert sha(path)==expected,('SHA differs',str(path))
        pins[str(path)]=expected;return path
    def read_json(path,expected):return json.loads(pin(path,expected).read_text())
    def arrays(directory,descriptor,label):
        name=descriptor['file'];assert Path(name).name==name
        path=pin(directory/name,descriptor['sha256']);assert path.stat().st_size==descriptor['bytes']
        with np.load(path,allow_pickle=False) as archive:
            assert set(archive.files)==set(descriptor['arrays']),(label,'keys')
            values={name:archive[name] for name in archive.files}
        details={}
        for name,value in values.items():
            actual=meta(value);expected=descriptor['arrays'][name]
            assert all(actual[k]==expected[k] for k in actual),(label,name,'descriptor')
            finite=int(np.count_nonzero(np.isfinite(value)))
            assert finite==value.size,(label,name,'nonfinite')
            for k,v in (('finite_elements',finite),('nonfinite_elements',value.size-finite)):
                if k in expected:assert expected[k]==v,(label,name,k)
            details[name]=dict(actual,finite=True)
        artifacts.append(dict(label=label,path=str(path),file_sha256=descriptor['sha256'],
            bytes=descriptor['bytes'],arrays=details))
        return values
    original=read_json(args.original_dir/'run.json',ORIGINAL_SHA)
    normal=read_json(args.v40_normal_dir/'run.json',NORMAL_SHA)
    export=read_json(args.v40_export_manifest,EXPORT_SHA)
    assert normal['export_manifest_sha256']==EXPORT_SHA
    assert normal['original_run_sha256']==export['original_run_sha256']==ORIGINAL_SHA
    original_manifest=read_json(args.original_dir/'input-manifest.json',original['input_manifest_sha256'])
    assert len(original_manifest['frames'])==192
    old_captures={v['frame_index']:v for v in original['update_captures']}
    exported={v['observation_id']:v for v in export['cases']}
    # The manifest is read before resolving old input/gold or normal output paths.
    for frame in (34,127,191):
        assert exported[frame]['original_inputs_artifact']==old_captures[frame]['input']
        assert exported[frame]['original_gold_artifact']==old_captures[frame]['gold']
        assert old_captures[frame]['same_call_original_Torch'] is True
    original_inputs={frame:arrays(args.original_dir,old_captures[frame]['input'],f'v39_obs{frame}_input')
                     for frame in (34,127,191)}
    results=[];cpu_inputs={};cpu_outputs={}
    for label,directory,expected in (('CPU_encoder_RKNN_Update',args.cpu_dir,CPU_SHA),
                                    ('RKNN_encoder_RKNN_Update',args.both_dir,BOTH_SHA)):
        run=read_json(directory/'run.json',expected)
        assert run['schema']=='v41_cached_undistort_online_RKNN_research_replay'
        assert run['status']=='failed' and 'offline_termination failed' in run['error'] and 'B1_D3_M16' in run['error']
        assert run['frames_requested']==run['frames_completed']==len(run['frames'])==192
        assert sum(bool(v['result']['fresh_pose_valid']) for v in run['frames'])==158
        assert 'termination' not in run and 'final_snapshot' not in run
        assert run['candidate_accepted'] is False and run['adopted'] is False
        manifest=read_json(directory/'input-manifest.json',run['input_manifest_sha256'])
        assert manifest==original_manifest and run['input_manifest_sha256']==original['input_manifest_sha256']
        assert run['policy_sha256']==original['policy_sha256'] and run['policy']==original['policy'] and run['config']==original['config']
        for i,(frame,source) in enumerate(zip(run['frames'],manifest['frames'])):
            result=frame['result']
            assert frame['index']==i and result['observation_id']==i
            assert frame['frame_id']==result['frame_id']==source['frame_id']
            assert frame['timestamp']==result['timestamp']==source['timestamp']
        snapshot=arrays(directory,run['snapshot_before_termination']['artifact'],label+'_before_termination')
        assert snapshot['poses'].dtype==snapshot['patches'].dtype==snapshot['net'].dtype==np.float32
        assert all(snapshot[k].dtype==np.int64 for k in ('ii','jj','kk','ii_inactive','jj_inactive','kk_inactive'))
        cache=run['online_metrics']['cache']
        assert run['online_metrics']['closed'] and cache['closed'] and cache['resident_contexts']==0
        assert cache['current_DMA_bytes']==cache['current_model_host_bytes']==0 and cache['cleanup_errors']==[]
        assert not cache['sessions'] and all(v['metrics']['context_destroyed'] and not v['metrics']['cleanup_errors'] for v in cache['histories'])
        assert run['encoder_metrics']['closed'] and run['cleanup_errors']==[] and run['affinity_restored']
        assert run['original_weights_before']==run['original_weights_after']==original['original_weights_before']
        assert len(run['original_weights_before'])==94 and run['original_weights_unchanged']
        assert run['sources_and_inputs_unchanged'] and run['source_pins']==run['source_hashes_after']
        if label=='RKNN_encoder_RKNN_Update':
            assert run['encoder_metrics']['context']['context_destroyed'] and not run['encoder_metrics']['cleanup_errors']
        captures=[]
        assert [v['frame_index'] for v in run['update_captures']]==[34,127,191]
        for capture in run['update_captures']:
            frame=capture['frame_index'];edges=capture['edges']
            assert capture['entire_active_graph'] and capture['same_call_original_Torch'] is False
            assert capture['reference_supplied'] is False and capture['output_backend']=='RKNN_complete_source_groups'
            assert 'gold' not in capture
            inputs=arrays(directory,capture['input'],label+f'_obs{frame}_input')
            output=arrays(directory,capture['output'],label+f'_obs{frame}_output')
            assert set(inputs)==INPUTS and set(output)==OUTPUTS
            for name,channels in (('net',384),('inp',384),('corr',882)):
                assert inputs[name].dtype==np.float32 and inputs[name].shape==(1,edges,channels)
            for name in ('ii','jj','kk'):assert inputs[name].dtype==np.int64 and inputs[name].shape==(edges,)
            for name,channels in (('net',384),('delta',2),('weight',2)):
                assert output[name].dtype==np.float32 and output[name].shape==(1,edges,channels)
            assert np.all((output['weight']>=0)&(output['weight']<=1))
            equality={k:equal(inputs[k],original_inputs[frame][k]) for k in INPUTS}
            captures.append(dict(frame_index=frame,edges=edges,all_saved_inputs_and_outputs_verified=True,
                input_bytes_equal_original_CPU_capture=equality,all_six_inputs_equal=all(equality.values()),
                old_same_call_gold_comparison_performed=False))
            if label=='CPU_encoder_RKNN_Update':cpu_inputs[frame]=inputs;cpu_outputs[frame]=output
        failed=run['online_metrics']['calls'][-1]
        assert failed['index']==203 and failed['status']=='failed_no_partial_fullgraph'
        failure_name=f'update-{failed["index"]:06d}-failed.json'
        failure=read_json(directory/'online-sdk'/failure_name,sha(directory/'online-sdk'/failure_name))
        assert failure==failed and 'B1_D3_M16' in failure['error']
        assert failure['route']=='full' and failure['fullgraph_returned'] is False
        assert failure['outputs_committed_by_callback'] is False
        failed_inputs=arrays(directory/'online-sdk',failure['failure_artifact'],label+'_EOF_failed_six_inputs')
        assert set(failed_inputs)==INPUTS
        e=failure['edges']
        for name,c in (('net',384),('inp',384),('corr',882)):
            assert failed_inputs[name].shape==(1,e,c) and failed_inputs[name].dtype==np.float32
        for name in ('ii','jj','kk'):assert failed_inputs[name].shape==(e,) and failed_inputs[name].dtype==np.int64
        assert all(equal(failed_inputs[name],snapshot[name]) for name in ('net','ii','jj','kk'))
        results.append(dict(label=label,run_sha256=expected,original_status_preserved=run['status'],
            original_error=run['error'],online_observations_verified=192,fresh_count_from_metadata=158,
            source_plan_identical_to_original_192=True,before_termination_arrays_verified=True,
            captures=captures,EOF_failure=dict(index=203,unknown_shape='B1_D3_M16',
                failure_json_bound_to_run_record=True,six_saved_inputs_verified=True,
                recurrent_and_topology_bytes_equal_before_termination=True,fullgraph_returned=False),
            metadata_only_lifecycle_checks=dict(contexts_closed=True,encoder_closed=True,original94_weights_stable=True,
                source_pins_stable=True,cleanup_errors=[],neural_weights_reexecuted=False),
            reported_SDK_activity=run['SDK_activity']))
    assert all(equal(cpu_inputs[34][k],original_inputs[34][k]) for k in INPUTS)
    gold=arrays(args.original_dir,old_captures[34]['gold'],'v39_obs34_same_call_original_Torch_gold')
    comparison={k:compare(cpu_outputs[34][k],gold[k]) for k in OUTPUTS}
    strict=all(v['gate_passed'] for v in comparison.values());assert strict is False
    for case in results[0]['captures']:
        if case['frame_index']==34:case['old_same_call_gold_comparison_performed']=True
    normal_equal=[]
    for call in normal['calls']:
        if call['observation_id']!=34:continue
        values=arrays(args.v40_normal_dir,call['full_order_artifact'],f'v40_obs34_repeat{call["repeat_index"]}_full_output')
        equality={k:equal(cpu_outputs[34][k],values[k]) for k in OUTPUTS}
        assert all(equality.values())
        normal_equal.append(dict(repeat_index=call['repeat_index'],all_three_outputs_bitwise_equal=True,arrays=equality))
    assert len(normal_equal)==3
    after={path:sha(path) for path in pins};assert after==pins
    report=dict(schema='v41_online_saved_array_integrity_audit',status='passed_saved_array_integrity_runs_still_failed',
        audit_script_sha256=sha(__file__),runs=results,artifacts=artifacts,
        actual_NPZ_artifacts_reread=len(artifacts),actual_arrays_verified=sum(len(v['arrays']) for v in artifacts),
        CPU_encoder_obs34=dict(all_six_inputs_bitwise_equal_original_CPU=True,
            saved_same_input_original_Torch_reference_only=True,original_Torch_forward_reexecuted=False,
            strict_FP32_gate_passed=False,comparison=comparison,v40_normal_output_bitwise_comparison=normal_equal),
        candidate_accepted=False,GT_read=False,real_forward_executed=False,SDK_executed=False,board_execution=False,
        source_and_array_pins=pins,source_hashes_after=after,all_audit_inputs_unchanged=True,
        limitations=['Both whole runs remain failed at the first offline EOF B1D3 call; the192 online prefix is not whole-run completion.',
            'Context destruction/source/94-weight stability are checks of pinned recorded metadata, not a rerun or independent board memory measurement.',
            'Obs127/191 feedback inputs differ from old captures; no mismatched-input gold comparison.',
            'Both-NPU captures have no same-call original-Torch teacher; old CPU gold is never used to validate them.',
            'Finite saved outputs and bitwise repeatability do not establish FP32 equivalence, GT accuracy, adoption or internal all-NPU placement.'])
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],NPZ=report['actual_NPZ_artifacts_reread'],arrays=report['actual_arrays_verified'],
        CPU_obs34_strict_FP32_gate_passed=strict,CPU_obs34_v40_all_three_repeats_bitwise_equal=True,
        output=str(args.output),output_sha256=sha(args.output))))


if __name__=='__main__':main()
