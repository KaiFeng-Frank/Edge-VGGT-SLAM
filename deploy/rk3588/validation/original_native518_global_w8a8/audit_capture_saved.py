#!/usr/bin/env python3
"""Reproduce v115 capture/FP32-gate checks using saved NumPy files only."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('/dev/shm/edge-vggt-v115-review')
CAPTURE_SHA='ea62a14c2bf219021d6d3b14647a4201773d3a05bc03c705cb0c0e87b7ac9ee3'
SOURCE_SHA='2fd3f55672de3a76376a712e707b202cad57e5b2dfbc2f2f605de4f57324857b'
CONTROL_SHA='e67fcc6b1226463529e37e310586dea0cfbf0ad3cb5eb4a5bd3c42f5b81afdbc'
CHECK_SHA='8a3b0c795758c9183c48795e421ee04a721547dcc4f3c1f81945edfe2df50f16'
QUALITY_SHA='37a7856a6037fe16b29dc601dcf488c12a513d457dcdca8bb2e747ed43f1f77c'
NATIVE_SHA='4ebbc4adaaed5e3c3cf8558b4947ec3fc090d56e82f8c61c9509951eb34dd35d'
CAL_SHA='a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
WEIGHTS_SHA='4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CHECKPOINT_SHA='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
SHAPE=(1,2082,1024)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def read(path,digest):
    assert sha(path)==digest, str(path)
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir',type=Path,default=ROOT/'capture-r2')
    p.add_argument('--control',type=Path,default=ROOT/'capture-control-r2/control.json')
    p.add_argument('--check-dir',type=Path,default=ROOT/'check-r3')
    p.add_argument('--capture-source',type=Path,default=ROOT/'capture_native_global0_inputs.py')
    p.add_argument('--quality',type=Path,default=ROOT/'quality-host-reference.json')
    p.add_argument('--native',type=Path,default=Path('/dev/shm/edge-vggt-v98-native518/reference-r1/capture.json'))
    p.add_argument('--calibration-metadata',type=Path,default=Path('/dev/shm/edge-vggt-v56/calibration-capture.json'))
    p.add_argument('--weights-metadata',type=Path,default=Path('/dev/shm/edge-vggt-v98-native518/weights-capture/capture.json'))
    p.add_argument('--output',type=Path,default=ROOT/'capture-saved-audit.json')
    a=p.parse_args()
    assert sha(a.capture_source)==SOURCE_SHA
    r=read(a.capture_dir/'result.json',CAPTURE_SHA); control=read(a.control,CONTROL_SHA)
    check=read(a.check_dir/'result.json',CHECK_SHA); q=read(a.quality,QUALITY_SHA); n=read(a.native,NATIVE_SHA)
    cal=read(a.calibration_metadata,CAL_SHA); weights=read(a.weights_metadata,WEIGHTS_SHA)
    assert r['status']=='completed_original_native518_global0_input_capture' and r['script_sha256']==SOURCE_SHA
    assert r['capture_revision']==2 and r['checkpoint']==dict(bytes=5026874952,sha256=CHECKPOINT_SHA)
    clean=('source_and_inputs_unchanged','model_state_unchanged','hooks_removed','affinity_restored','environment_restored',
        'sys_path_restored','all_NPU_contexts_closed','resource_counts_balanced','complete_expected_execution',
        'checkpoint_verified_before_deserialization','checkpoint_stable_stat_unchanged','checkpoint_final_SHA_verified',
        'calibration_evaluation_disjoint','heldout_not_in_calibration','quality_and_fixed_PNG_disjoint')
    assert all(r[k] is True for k in clean) and not r['cleanup_errors'] and not r['changed_sources_or_inputs']
    assert r['original_parameters']==next(x['parameters'] for x in weights['blocks'] if x['role']=='global' and x['index']==0)
    assert r['original_CPU_global0_forward_count']==6 and r['original_CPU_reference_control_count']==1 and r['original_CPU_reference_control_byte_equal']
    assert r['checkpoint_final_sha256']==CHECKPOINT_SHA
    stable=('st_dev','st_ino','st_mode','st_nlink','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns')
    initial=r['checkpoint_initial_constructor_stat']; rows=r['checkpoint_stat_observations']
    assert len(rows)==2 and [x['phase'] for x in rows]==['after_all_NPU_contexts_closed','finally']
    for x in rows:
        assert x['initial']==initial and all(initial[k]==x['current'][k] for k in stable)
        differing=[k for k in initial if initial[k]!=x['current'][k]]
        assert differing==x['differing_fields'] and x['stable_fields_equal']
        assert x['atime_changed']==(initial['st_atime_ns']!=x['current']['st_atime_ns']) and x['only_atime_diff']==(differing==['st_atime_ns'])
    assert all(initial[k]==r['checkpoint_stat_after_final_SHA'][k] for k in stable)
    records=r['records']; ids=['calib00','calib01','calib02','calib03','heldout00','heldout01']
    assert [x['id'] for x in records]==ids and [x['role'] for x in records]==['calibration']*4+['heldout']*2
    selected=[cal['samples'][i] for i in (0,1,2,3,5,6)]
    for row,old in zip(records,selected):
        assert row['source_pair_indices']==old['indices'] and row['source_images']==old['images']
        assert row['preprocessed_pixels']['shape']==[2,3,392,518] and row['input']['raw_sha256']==row['frame0_output_raw_sha256']
    cpng={x['sha256'] for row in records[:4] for x in row['source_images']}; epng={x['sha256'] for row in records[4:] for x in row['source_images']}
    excluded={x['sha256'] for x in q['source_images']}|{x['sha256'] for x in n['source_images']}
    assert len(cpng)==8 and len(epng)==4 and not cpng&epng and not (cpng|epng)&excluded
    assert len({x['input']['raw_sha256'] for x in records})==6 and len({x['preprocessed_pixels']['raw_sha256'] for x in records})==6
    import numpy as np
    arrays=[]
    def array(folder,meta,tag,norm=False):
        path=folder/meta['file']; assert Path(meta['file']).name==meta['file'] and path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256']
        x=np.load(path,allow_pickle=False)
        assert x.shape==SHAPE and x.dtype==np.float32 and x.flags.c_contiguous and x.nbytes==meta['tensor_bytes']==8527872 and np.isfinite(x).all()
        raw=hashlib.sha256(x.tobytes(order='C')).hexdigest(); assert raw==meta['raw_sha256']
        if norm:
            assert float(x.min())==meta['minimum'] and float(x.max())==meta['maximum']
            assert math.isclose(float(np.linalg.norm(x.astype(np.float64).ravel())),meta['l2_norm'],rel_tol=1e-12,abs_tol=1e-9)
        arrays.append(dict(tag=tag,file=meta['file'],bytes=meta['bytes'],sha256=meta['sha256'],raw_sha256=raw)); return x
    refs={}
    for i,row in enumerate(records):
        array(a.capture_dir,row['input'],row['id']+'_input',True)
        assert row['reference']['saved']==(i>=4)
        if i>=4: refs[row['id']]=array(a.capture_dir,row['reference'],row['id']+'_CPU_reference',True)
        else: assert 'file' not in row['reference'] and row['reference']['finite']
    assert len(arrays)==8 and {x['file'] for x in arrays}=={p.name for p in a.capture_dir.glob('*.npy')}
    dataset=r['dataset']; dp=a.capture_dir/dataset['file']; assert dp.stat().st_size==dataset['bytes'] and sha(dp)==dataset['sha256']
    lines=dp.read_text().splitlines(); assert len(lines)==dataset['line_count']==4 and dataset['paths_are_absolute']
    assert lines==['/tmp/vggt-v115-capture-r2/'+row['input']['file'] for row in records[:4]]
    assert sum(p.stat().st_size for p in a.capture_dir.iterdir() if p.is_file())==r['new_capture_bytes']<70000000
    counters=('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')
    assert len(r['resources'])==5 and [x['runs_completed'] for x in r['resources']]==[12,12,12,12,6]
    for x in r['resources']:
        assert x['closed'] and x['resource_counts_balanced'] and not x['live_contexts'] and not x['live_output_buffers'] and not x['cleanup_errors']
        assert x['core_mask']==7 and x['init_flags']==0 and x['contexts_created']==x['contexts_destroyed']==1
        assert x['runs_completed']==x['output_buffers_acquired']==x['output_buffers_released']
    totals={k:sum(x[k] for x in r['resources']) for k in counters}
    assert totals==r['SDK_totals']==dict(contexts_created=5,contexts_destroyed=5,runs_completed=54,output_buffers_acquired=54,output_buffers_released=54)
    assert control['child_returncode']==0 and not control['restore_errors'] and control['frequency_governor_limits_restored']
    assert control['baseline']==control['restored'] and control['restored']['cur_freq']=='300000000' and control['seconds']<180
    co=control['observations']; ro=r['platform_observations']; assert co and ro
    assert all(x['actual_frequency_hz']==1000000000 for x in co) and all(x['npu_hz']==1000000000 for x in ro)
    peak=max(v for x in co+ro for v in x['temperatures_c'].values()); assert peak<75
    assert check['status']=='verified_original_global0_FP32_heldout_not_quantized' and check['verify_only'] and not check['calibration_executed']
    assert check['capture']['sha256']==CAPTURE_SHA and check['capture_source']['sha256']==SOURCE_SHA and check['original_parameters']==r['original_parameters']
    assert check['source_and_inputs_unchanged'] and check['affinity_restored'] and check['environment_restored'] and check['cwd_restored'] and not check['cleanup_errors']
    fixed=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5); assert check['fixed_gate']==fixed
    gates=[]
    for entry,row in zip(check['FP32_ONNX_checks'],records[4:]):
        assert entry['id']==row['id'] and entry['input_sha256']==row['input']['sha256'] and entry['reference_sha256']==row['reference']['sha256']
        x=array(a.check_dir,entry['output'],row['id']+'_FP32_ONNX'); y=refs[row['id']]
        d=x.astype(np.float64)-y.astype(np.float64); relative=float(np.linalg.norm(d.ravel())/np.linalg.norm(y.astype(np.float64).ravel())); maximum=float(np.abs(d).max())
        outside=int(np.count_nonzero(np.abs(d)>1e-5+1e-4*np.abs(y.astype(np.float64)))); strict=bool(np.allclose(x,y,rtol=1e-4,atol=1e-5))
        assert strict and outside==0 and relative<=1e-5 and maximum<=1e-3 and entry['comparison']['fixed_gate_passed']
        assert math.isclose(relative,entry['comparison']['relative_l2'],rel_tol=1e-11,abs_tol=1e-14) and maximum==entry['comparison']['max_abs']
        assert (x.tobytes()==y.tobytes())==entry['comparison']['byte_equal']
        gates.append(dict(id=row['id'],relative_l2=relative,max_abs=maximum,outside=outside,strict_allclose=strict,byte_equal=x.tobytes()==y.tobytes()))
    assert len(gates)==2
    report=dict(schema='v115_independent_saved_capture_and_original_FP32_gate_audit',status='passed_saved_files_and_receipts',
        auditor_sha256=sha(__file__),saved_NumPy_only=True,NN_SDK_ORT_GPU_or_board_execution=False,
        capture_receipt_sha256=CAPTURE_SHA,capture_source_sha256=SOURCE_SHA,control_sha256=CONTROL_SHA,FP32_check_receipt_sha256=CHECK_SHA,
        capture_arrays_verified=8,FP32_ONNX_arrays_verified=2,verified_arrays=arrays,
        calibration_cases=4,heldout_cases=2,source_PNGs=12,all_PNG_and_activation_partitions_disjoint=True,
        source_pair_indices=[x['source_pair_indices'] for x in records],dataset_rows=4,capture_bytes=r['new_capture_bytes'],
        SDK_totals=totals,all_contexts_and_buffers_closed=True,CPU_reference_calls_reported=6,old_fixed_CPU_byte_control_reported=True,
        control_seconds=control['seconds'],capture_wall_seconds_not_performance=r['elapsed_seconds'],capture_peak_RSS_MiB=r['peak_process_RSS_MiB'],
        controller_observations=len(co),capture_observations=len(ro),all_observed_NPU_hz=1000000000,maximum_observed_temperature_c=peak,restored_NPU_hz=300000000,
        stable_checkpoint_fields_unchanged=True,R2_differing_stat_fields=[x['differing_fields'] for x in rows],
        final_full_original_checkpoint_SHA_reported=True,FP32_ONNX_checks=gates,fixed_gate=fixed,
        candidate_accepted=False,full_model_quality_or_realtime_accepted=False,
        limits=['Four calibration CPU outputs are metadata-only; only two heldout CPU payloads can be independently rechecked here.',
            'Reference computation and final5GB checkpoint SHA are recorded board claims corroborated by source/receipt checks; this auditor does not execute models or reread the board checkpoint.',
            'R2 records no stat difference, including atime. This cannot establish which exact stat field differed in R1.',
            'Original standalone frame0 boundary capture is deployment-derived, not certified internal retained-group activation byte identity; no full-model acceptance.'])
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(report=str(a.output),sha256=sha(a.output),arrays=10,peak_temperature_c=peak,capture_bytes=r['new_capture_bytes'],status=report['status'])))


if __name__=='__main__': main()
