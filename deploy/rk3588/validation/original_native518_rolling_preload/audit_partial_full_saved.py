#!/usr/bin/env python3
"""Saved-only audit of the interrupted R123 full-model trial.

Verify the completed first A prediction and preserve the abrupt B failure.
No inference, preprocessing, SDK, GPU or board execution; no B timing claim.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT=Path('/dev/shm/edge-vggt-v123-review')
REF=Path('/path/to/research-artifacts/v111-native518-trajectory-quality-r1/batch0-r1')
RESULT_SHA='f0e3e10b3e81521be00496971edfb46719f67f41c7cb3bebea3d60e2d2eef1b0'
CONTROL_SHA='b9697cd50b0873b9a5e33d8a9a2dd6fd6250686b19bbcd21dc191d8cf3d5ceff'
REF_SHA='2eb6f63d2c832c92631a55e527f4b49e3bf0b197685f3bdea8f76008aada1081'
MANIFEST_SHA='9a65445736f7c5a6cfa8c66053c746c05a4e9e3c313b218562ef7f3e72f0f84a'
OUTPUTS=('pose_enc','extrinsic','intrinsic','depth','depth_conf')
COLD=dict(contexts_created=10,contexts_destroyed=10,runs_completed=18,
    output_buffers_acquired=23,output_buffers_released=23)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def raw(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--reference',type=Path,default=REF)
    p.add_argument('--output',type=Path,default=ROOT/'partial-full-r1-saved-audit.json')
    a=p.parse_args();files=[]
    def record(path,digest=None):
        actual=sha(path);require(digest is None or actual==digest,'Pinned identity differs: '+str(path))
        files.append(dict(file=str(path),bytes=path.stat().st_size,sha256=actual));return path
    def receipt(path,digest=None):return json.loads(record(path,digest).read_text())
    def load(base,row):
        require(Path(row['file']).name==row['file'],'NPY must be relative basename')
        path=record(base/row['file'],row['sha256']);v=np.load(path,allow_pickle=False)
        require(path.stat().st_size==row['bytes'] and list(v.shape)==row['shape'] and str(v.dtype)==row['dtype'] and
            raw(v)==row['raw_sha256'] and v.dtype==np.float32 and np.isfinite(v).all(),'NPY size/shape/dtype/raw/finite differs')
        files[-1].update(raw_sha256=raw(v),shape=list(v.shape),dtype=str(v.dtype),finite=True);return v
    r=receipt(a.root/'full-r1/result.json',RESULT_SHA);c=receipt(a.root/'control-r1/control.json',CONTROL_SHA)
    old=receipt(a.reference/'result.json',REF_SHA);ref=old['pairs'][0]
    require(r['schema']=='v123_original_native518_full_model_rolling_preload_ABBA' and r['status']=='running' and
        r['stage']=='complete_original_prediction_start' and len(r['calls'])==1 and r['calls'][0]['index']==0 and
        r['calls'][0]['route']=='A' and c['child_returncode']==-7,'Exact interrupted first-B trial required')
    require(r['reference_receipt_sha256']==REF_SHA and r['reference_output_metadata']==ref['outputs'] and
        r['source_images']==ref['source_images'] and r['reference_input_raw_sha256']==ref['prepared_pair_raw_sha256'] and
        r['source_indices']==ref['source_indices']==[32,56] and r['shape']==[2,3,392,518] and
        r['core_mask']==7 and r['previous_tail_cache_enabled'] is False and not r['candidate_accepted'],
        'Exact unchanged native cold reference/input required')
    src=a.root/'executed-source-r1'
    require(receipt(src/'sources.json',MANIFEST_SHA)==r['source_bundle'] and r['source_manifest_sha256']==MANIFEST_SHA,
        'Executed source manifest differs')
    for name,meta in r['source_bundle'].items():
        path=record(src/name,meta['sha256']);require(path.stat().st_size==meta['bytes'],'Executed source size differs')
        ast.parse(path.read_text())
    row=r['calls'][0];ec=row['executor_call'];require(set(row['outputs'])==set(OUTPUTS),'All5 first-A predictions required')
    require(ec['status']=='completed' and ec['complete_expected_execution'] and ec['resources_balanced'] and
        ec['source_and_inputs_unchanged'] and not ec['cleanup_errors'] and not ec['tail_cache_hit'] and
        ec['encoder_frame_indices_computed']==[0,1] and ec['requested_core_mask']==7 and ec['SDK_totals']==COLD and
        row['original_input_raw_sha256']==ec['input_raw_sha256']==ref['prepared_pair_raw_sha256'] and
        ec['input_frame_raw_sha256']==ref['prepared_frame_raw_sha256'],'Completed first-A actual input/resource proof differs')
    comparisons={}
    for key in OUTPUTS:
        v=load(a.root/'full-r1',row['outputs'][key]);z=load(a.reference,ref['outputs'][key])
        equal=v.tobytes()==z.tobytes();require(equal and row['byte_equal_frozen_full_model'][key],'First-A reference bytes differ')
        comparisons[key]=dict(array_bytes_equal=True,raw_sha256=raw(v),shape=list(v.shape),dtype='float32',
            file_sha256_equal=row['outputs'][key]['sha256']==ref['outputs'][key]['sha256'])
    require({p.name for p in (a.root/'full-r1').glob('*.npy')}=={v['file'] for v in row['outputs'].values()},
        'Only5 first-A arrays should be preserved; no B output exists')
    owners=ec['resources'];require(set(owners)=={'dino0','dino1','dino2','dino3','aggregator','heads'},'First-A owners differ')
    expected=[(owners['dino'+str(i)],1,2,2) for i in range(4)]
    expected += [(owners['aggregator'],4,4,9)]
    expected += [(v,1,runs,buffers) for v,runs,buffers in zip(owners['heads']['contexts'],(4,2),(4,2))]
    require(len(owners['heads']['contexts'])==2,'Original camera and depth contexts required')
    for value,contexts,runs,buffers in expected:
        require(value['contexts_created']==value['contexts_destroyed']==contexts and value['runs_completed']==runs and
            value['output_buffers_acquired']==value['output_buffers_released']==buffers and not value['live_contexts'] and
            not value['live_output_buffers'] and not value['cleanup_errors'],'First-A SDK owner not balanced')
    require(r['DINO_layers_per_frame']==24 and r['aggregator_blocks_per_pair']==48 and r['camera_refinements']==4 and
        r['DPT_frames']==2 and r['original_pose_codec'] and r['reference_intermediate_substituted'] is False,
        'Original complete first-A model scope differs')
    require(math.isclose(row['image_read_to_prediction_seconds'],row['preprocessing_seconds']+row['complete_prediction_seconds'],
        rel_tol=1e-12),'First-A timing accounting differs')
    require(c['status']=='failed' and c['restore_errors']==[] and c['baseline']==c['restored'] and
        c['frequency_governor_limits_restored'] and c['restored']['cur_freq']=='300000000' and
        all(v['actual_frequency_hz']==1000000000 for v in c['observations']),'External failure/frequency restoration differs')
    maximum=max(t for v in c['observations'] for t in v['temperatures_c'].values())
    logpath=record(a.root/'control-r1/worker.log');log=logpath.read_text()
    require('errno: 12' in log and 'size: 92124480, flags: 0x2' in log,'Actual saved SDK allocation failure missing')
    starts=[];groups={0:[],1:[]}
    for line in log.splitlines():
        if not line.startswith('{'):continue
        value=json.loads(line)
        if value.get('stage')=='complete_original_prediction_start':starts.append((value['index'],value['route']))
        if value.get('stage')=='fresh_native_DINO_group':groups[value['call']].append(value['group'])
    require(starts==[(0,'A'),(1,'B')] and groups=={0:[0,1,2,3],1:[0,1,2,3]},'Actual saved worker progress differs')
    tracepath=record(a.root/'board-dmesg-after-r1.txt');trace=tracepath.read_text()
    new=[line for line in trace.splitlines() if line.startswith('[346650.')]
    alloc=next(line for line in new if 'page allocation failure' in line)
    dma=next(line for line in new if 'DMA free:' in line);normal=next(line for line in new if 'Normal free:' in line)
    dma_values=re.search(r'DMA free:(\d+)kB min:(\d+)kB',dma);normal_free=re.search(r'Normal free:(\d+)kB',normal)
    require('GFP_DMA32' in alloc and any('rknpu_gem_object_create' in line for line in new) and
        dma_values is not None and normal_free is not None,'Saved new rknpu allocation trace missing')
    result=dict(schema='v123_independent_saved_partial_full_model_failure_audit',status='verified_partial_failure_saved_evidence',
        auditor_sha256=sha(__file__),execution='Saved JSON/NumPy/text/AST only; no inference/preprocessing/SDK/GPU/board calls',
        planned_schedule='ABBA',completed_prediction_calls=1,completed_A_calls=1,completed_B_calls=0,ABBA_completed=False,
        checked_actual_NPYs=5,first_A_all5_outputs_byte_equal_frozen_R111=True,output_comparisons=comparisons,
        first_A_prepared_pair_raw_sha256=ref['prepared_pair_raw_sha256'],first_A_original_source_images=ref['source_images'],
        first_A_cold_execution=dict(DINO_layers_per_frame=24,frames_computed=2,aggregator_blocks=48,camera_refinements=4,
            DPT_frames=2,original_codec=True,SDK_totals=COLD,all_completed_A_SDK_owners_balanced=True),
        first_A_timing={k:row[k] for k in ('preprocessing_seconds','complete_prediction_seconds','image_read_to_prediction_seconds','save_and_compare_seconds')},
        failed_B_evidence=dict(started=True,original_DINO_group_progress=[0,1,2,3],complete_prediction_saved=False,
            final_output_payload_count=0,child_returncode=-7,termination='SIGBUS',SDK_allocation_errno=12,SDK_failed_allocation_bytes=92124480,
            worker_final_SDK_cleanup_verified=False,worker_sampler_join_verified=False,worker_final_pin_and_environment_restore_verified=False),
        raw_worker_receipt_status_preserved=r['status'],final_executor_resource_receipt_present='executor_resources' in r,
        B_performance_available=False,performance_comparison_or_candidate_acceptance=False,
        kernel_allocation_evidence=dict(boot_time_prefix='346650',GFP_DMA32=True,rknpu_gem_path_observed=True,
            DMA_free_KiB=int(dma_values[1]),DMA_min_KiB=int(dma_values[2]),Normal_free_KiB=int(normal_free[1]),
            note='Saved allocation evidence supports DMA-zone pressure while Normal pages remained; exact SIGBUS stack/instruction was not captured'),
        external_control=dict(child_returncode=-7,observed_NPU_hz=1000000000,restored_NPU_hz=300000000,
            exact_frequency_policy_restored=True,max_sampled_temperature_c=maximum,seconds=c['seconds']),
        limits=['Only the first completed sequential A was verified; no completed rolling B or matched ABBA comparison',
            'The abrupt signal bypassed Python finally; empty progressive cleanup_errors is not a cleanup guarantee',
            'Controller frequency restoration does not establish failed-worker SDK destroy/release balances',
            'Pixel-byte match comes from executed frozen preprocessing receipts; no host preprocessing was rerun',
            'R111 is the original native NPU execution reference, not original CPU FP32 or trajectory quality acceptance',
            'No full-model timing benefit, P95, fresh20Hz, maprate,30min or8GB acceptance'],files=files)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],completed_A=1,completed_B=0,arrays=5,
        A_seconds=row['complete_prediction_seconds'],child_returncode=-7,worker_cleanup_claimed=False)))


if __name__=='__main__':main()
