#!/usr/bin/env python3
"""Audit saved original attention capture only; no Torch, model, ORT or SDK."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

CAPTURE_SOURCE_SHA = 'c307091ec55daa32d3cb1fd53cef3f931bd50e97feb6f1f63894a2eecf959a5c'
TAIL_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
CHECKPOINT = {'bytes':5026874952,'sha256':'d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def metadata(value):
    value=np.ascontiguousarray(value)
    return {'shape':list(value.shape),'dtype':str(value.dtype),'tensor_bytes':value.nbytes,
        'finite':bool(np.isfinite(value).all()),'raw_sha256':hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-dir',type=Path,required=True)
    p.add_argument('--capture-sha256',required=True)
    p.add_argument('--original-tail-receipt',type=Path,required=True)
    p.add_argument('--calibration-receipt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.capture_dir.resolve(strict=True)
    require(sha(root/'capture.json')==args.capture_sha256,'Capture receipt SHA differs')
    require(sha(args.original_tail_receipt)==TAIL_SHA,'Original tail receipt SHA differs')
    require(sha(args.calibration_receipt)==CALIBRATION_SHA,'Original calibration receipt SHA differs')
    cap=json.loads((root/'capture.json').read_text())
    tail=json.loads(args.original_tail_receipt.read_text());calib=json.loads(args.calibration_receipt.read_text())
    require(cap['schema']=='v83_original_VGGT_attention_tail_capture' and cap['status']=='completed_original_FP32_attention_tail_capture','Capture status/schema differs')
    require(cap['script_sha256']==CAPTURE_SOURCE_SHA and cap['checkpoint']==CHECKPOINT,'Original capture source/checkpoint differs')
    require(cap['original_tail_capture_sha256']==TAIL_SHA and cap['source_calibration_sha256']==CALIBRATION_SHA,'Original input lineage differs')
    require(cap['original_model_only'] and cap['actual_inference_on_board_only'] and not cap['NPU_or_cache_installed']
        and not cap['full_VGGT_forward_executed'] and not cap['DINO_or_heads_executed']
        and not cap['ORT_or_SDK_or_compiler_or_Solver_executed'],'Capture execution boundary differs')
    require(all(cap[k] for k in ('hooks_removed','model_state_unchanged','checkpoint_stat_unchanged',
        'source_and_inputs_unchanged','environment_restored','affinity_restored')) and cap['cleanup_errors']==[],'Capture cleanup/identity differs')
    require(cap['original_parameters_verified']==36 and cap['original_block_forward_count']==4,'Original parameter/forward count differs')
    expected_sources={n:d for n,d in calib['source_hashes'].items() if n.startswith('third_party/vggt/vggt/')}
    require(cap['source_hashes']==expected_sources,'Original source pins differ')
    require([c['tag'] for c in cap['cases']]==['v53','v56-calib00'],'Real case order differs')
    contract=cap['attention_contract']
    require(contract['heads']==16 and contract['head_dim']==64 and contract['head_scale']==.125
        and contract['q_norm_eps']==1e-5 and contract['k_norm_eps']==1e-5
        and contract['original_SDPA_scale_argument'] is None and contract['mask'] is None
        and contract['dropout_p']==0 and not contract['is_causal'] and contract['tensor_dtype']=='float32'
        and contract['rope_frequency']==100 and contract['rope_scaling_factor']==1 and contract['rope_component_dimension']==32,'Original attention contract differs')
    rows={role:next(r for r in tail['blocks'] if r['index']==0 and r['role']==role) for role in ('frame','global')}
    for role,row in rows.items():
        require(set(cap['parameters'][role])==set(row['parameters']) and len(row['parameters'])==18,'36 original parameter identities differ')
        for name,item in row['parameters'].items():
            actual=cap['parameters'][role][name]
            require(all(actual[k]==item[k] for k in ('shape','dtype','tensor_bytes','raw_sha256')) and actual['finite'],'Original parameter metadata differs')
    g=lambda index:next(g for g in calib['groups']['aggregator'] if g['index']==index)
    sample=lambda index:next(s for s in g(index)['samples'] if s['sample_id']=='calib00')
    artifacts=[];cases=[]
    def load(item,shape,dtype='float32'):
        path=(root/item['file']).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Saved artifact file/path identity differs')
        a=np.load(path,allow_pickle=False);m=metadata(a)
        require(m['shape']==list(shape) and m['dtype']==dtype and m['finite'] and all(m[k]==item[k] for k in m),'Saved artifact array metadata differs')
        artifacts.append({'file':item['file'],'bytes':item['bytes'],'sha256':item['sha256'],'raw_sha256':m['raw_sha256']})
        return np.ascontiguousarray(a)
    for case in cap['cases']:
        tag=case['tag'];record={'tag':tag,'roles':{}}
        input_item=rows['frame']['input'] if tag=='v53' else sample(0)['artifact']
        require(case['original_frame_input']==input_item,'Original real frame0 input metadata differs')
        images=tail['source_images'] if tag=='v53' else next(s for s in calib['samples'] if s['sample_id']=='calib00')['images']
        require(case['source_images']==images,'Original image provenance differs')
        for role in ('frame','global'):
            b,n=(2,53) if role=='frame' else (1,106)
            row=case['roles'][role];items=row['artifacts'];arrays={}
            expected={'qkv_input','qkv_output','q_pre_norm','k_pre_norm','q_post_norm','k_post_norm',
                'proj_input','attention_output','v','q_post_rope','k_post_rope','attention_tail','block_output',
                'positions','q_norm_weight','q_norm_bias','k_norm_weight','k_norm_bias',
                'rope_cos_table','rope_sin_table','cos_y','cos_x','sin_y','sin_x'}
            require(set(items)==expected and row['block_index']==0,'Original hooked array set differs')
            for name,item in items.items():
                shape=(b,n,3072) if name=='qkv_output' else (b,n,1024) if name in ('qkv_input','proj_input','attention_output','block_output') else (b,n,2) if name=='positions' else (64,) if name.endswith('_weight') or name.endswith('_bias') else (9,32) if name.endswith('_table') else (b,1,n,32) if name in ('cos_y','cos_x','sin_y','sin_x') else (b,16,n,64)
                arrays[name]=load(item,shape,'int64' if name=='positions' else 'float32')
            q,k,v=arrays['qkv_output'].reshape(b,n,3,16,64).transpose(2,0,3,1,4)
            require(all(a.tobytes()==np.ascontiguousarray(z).tobytes() for a,z in ((arrays['q_pre_norm'],q),(arrays['k_pre_norm'],k),(arrays['v'],v))),'Original QKV split bytes differ')
            recovered=arrays['proj_input'].reshape(b,n,16,64).transpose(0,2,1,3)
            require(arrays['attention_tail'].tobytes()==np.ascontiguousarray(recovered).tobytes(),'Original pre-proj tail layout bytes differ')
            require(items['positions']['raw_sha256']==rows[role]['positions']['raw_sha256'],'Original RoPE position bytes differ')
            for coordinate,axis in ((0,'y'),(1,'x')):
                for trig in ('cos','sin'):
                    table=arrays['rope_'+trig+'_table']
                    require(items['rope_'+trig+'_table']['raw_sha256']==tail['rope_'+trig]['raw_sha256'],'Original RoPE source table bytes differ')
                    require(arrays[trig+'_'+axis].tobytes()==np.ascontiguousarray(table[arrays['positions'][...,coordinate]][:,None,:,:]).tobytes(),'Original RoPE indexed table bytes differ')
            for name in ('q_norm_weight','q_norm_bias','k_norm_weight','k_norm_bias'):
                param='attn.'+name.replace('_weight','.weight').replace('_bias','.bias')
                require(items[name]['raw_sha256']==rows[role]['parameters'][param]['raw_sha256'],'Original norm parameter bytes differ')
            reference=rows[role]['output'] if tag=='v53' else sample(1)['artifact'] if role=='global' else None
            require(row['original_block_reference']==reference,'Original block reference identity differs')
            if reference is not None:
                # The pinned earlier original CPU receipt fixes the exact reference
                # bytes. Matching raw SHA/dtype/size proves zero error after the
                # original global reshape; no new model calculation is performed.
                require(items['block_output']['raw_sha256']==reference['raw_sha256']
                    and items['block_output']['tensor_bytes']==reference['tensor_bytes']
                    and items['block_output']['dtype']==reference['dtype'],'Original saved block reference bytes differ')
                expected_check={'rtol':1e-4,'atol':1e-5,'strict_passed':True,'outside_tolerance':0,
                    'byte_equal':True,'max_abs_error':0.0,'relative_l2':0.0}
                require(row['reference_check']==expected_check,'Captured original strict reference metric differs')
            else: require(row['reference_check'] is None,'v56 frame0 has no preexisting block reference')
            if role=='frame':require(row['original_input_raw_sha256']==input_item['raw_sha256'],'Frame input raw SHA differs')
            else:require(row['original_input_raw_sha256']==case['roles']['frame']['artifacts']['block_output']['raw_sha256'],'Original frame-to-global view bytes differ')
            record['roles'][role]={'artifacts_verified':len(items),'QKV_split_byte_equal':True,
                'pre_proj_layout_byte_equal':True,'positions_and_rope_lookup_byte_equal':True,
                'old_original_block_reference_byte_equal':None if reference is None else True,
                'old_original_block_reference_strict_passed':None if reference is None else True}
        cases.append(record)
    report={'schema':'v83_original_attention_capture_saved_audit','status':'passed_saved_array_identity_audit',
        'capture_sha256':args.capture_sha256,'capture_script_sha256':CAPTURE_SOURCE_SHA,
        'auditor_source_sha256':sha(__file__),'original_tail_sha256':TAIL_SHA,'original_calibration_sha256':CALIBRATION_SHA,
        'cases':cases,'NPY_files_verified':len(artifacts),'total_NPY_bytes':sum(a['bytes'] for a in artifacts),
        'artifacts':artifacts,'original_parameters_verified':36,'original_block_forwards':4,
        'old_CPU_block_references_byte_equal':3,'all_hooks_state_inputs_sources_environment_affinity_restored':True,
        'cleanup_errors':[],'host_model_ORT_SDK_inference_executed':False,
        'limits':['QKNorm, RoPE and SDPA were captured on board; this audit does not recompute these model operations.',
                  'Three old block references verified by matching raw bytes to pinned original CPU receipt hashes.',
                  'Attention tails and v56 frame0 block output are new original CPU references, not earlier independent references.',
                  'No GPU candidate equivalence, model trajectory quality or inference performance acceptance.']}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','NPY_files_verified','total_NPY_bytes','old_CPU_block_references_byte_equal')}))


if __name__=='__main__':main()
