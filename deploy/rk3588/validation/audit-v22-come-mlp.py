#!/usr/bin/env python3
"""NumPy-only independent audit of frozen v22 saved arrays; no model inference."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture',type=Path,required=True)
parser.add_argument('--probe',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): parser.error('Output already exists; prior audits are never overwritten')
CAP=args.capture.resolve()
PROBE=args.probe.resolve()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a,b): return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
checks=[]
def check(name,condition):
    checks.append(name)
    assert bool(condition),name
check('capture_pin',sha(CAP/'capture.json')=='3664f5c932602a3c6538ea6013ea0389625c42bdcb7dcf10a9fbceb9304abeb1')
check('probe_pin',sha(PROBE/'probe.json')=='5abe4899111361fa3f0ba4b1fab78e3c15e984dbe358e2de05a001e0e8936f91')
c=json.loads((CAP/'capture.json').read_text());p=json.loads((PROBE/'probe.json').read_text())
for directory,receipt in ((CAP,c),(PROBE,p)):
    for name,info in receipt['artifacts'].items():
        check('artifact_'+directory.name+'/'+name,sha(directory/name)==info['sha256'])
        check('size_'+directory.name+'/'+name,(directory/name).stat().st_size==info['bytes'])
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
a=arrays(CAP/'activations.npz');w=arrays(CAP/'mlp-state.npz')
s=arrays(PROBE/'scene-plan.npz');o=arrays(PROBE/'outputs.npz')
for key in ('confidence','group_means'):
    bits=s[key+'_bf16_bits']
    check(key+'_bit_encoding',bits.dtype==np.uint16 and same((bits.astype(np.uint32)<<16).view(np.float32),s[key]))
conf=s['confidence'].reshape(2,18,24)
# Accumulate exact represented BF16 values in float64, then round nearest-even
# to BF16. This is saved-array arithmetic, never a predictor/model invocation.
means=conf.reshape(2,6,3,8,3).transpose(0,1,3,2,4).reshape(2,6,8,9).astype(np.float64).mean(axis=-1).astype(np.float32)
u=means.view(np.uint32)
rounded=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
check('independent_BF16_group_mean',same(rounded,s['group_means_bf16_bits']))
median=np.quantile(s['group_means'].astype(np.float32).ravel(),.5).astype(np.float32)
check('scene_median',float(median)==float(s['median_threshold'].item()))
mask=s['group_means']<median
check('strict_actual_mask',same(mask,s['merge_mask']))
check('no_forced_mask_count',p['plan']['compact_rows']==874-8*int(mask.sum()))
rev=np.empty((2,437),np.int64);reps=[];mins=[];mouts=[];offsets=[0];weights=[];cursor=0
for view in range(2):
    for j in range(5):
        rev[view,j]=cursor;reps.append(view*437+j);weights.append(1.);cursor+=1
    for gy in range(6):
        for gx in range(8):
            indices=np.array([5+(gy*3+y)*24+gx*3+x for y in range(3) for x in range(3)])
            if mask[view,gy,gx]:
                rev[view,indices]=cursor;reps.append(view*437+int(indices[0]));weights.append(9.)
                mins.append(view*437+indices);mouts.append(cursor);cursor+=1
            else:
                rev[view,indices]=np.arange(cursor,cursor+9);reps.extend((view*437+indices).tolist());weights.extend([1.]*9);cursor+=9
    offsets.append(cursor)
for key,value in [('reverse_indices',rev),('view_offsets',np.array(offsets,np.int64)),('representative_indices',np.array(reps,np.int64)),('mean_input_indices',np.array(mins,np.int64)),('mean_output_indices',np.array(mouts,np.int64)),('token_weights',np.array(weights,np.float32))]:
    check('independent_'+key,same(s[key],value))
pixelmask=np.repeat(np.repeat(mask,3,axis=1),3,axis=2)
check('all_merged_patch_rows',same(pixelmask,s['merged_patch_mask']))
x=a['norm2'].reshape(874,1024)
compact=x[reps].copy()
compact[mouts]=x[np.array(mins)].mean(axis=1)
check('compact_input_means_strict_FP32',np.allclose(compact.reshape(1,cursor,1024),o['merged_compact_input'],rtol=1e-4,atol=1e-5))
check('split_restores_repeated_outputs_exact',same(o['merged_compact_output'][0,rev.ravel()].reshape(1,874,1024),o['merged_mlp_output']))
for mode in ('original','merged'):
    check(mode+'_original_LS2',same(o[mode+'_mlp_output']*w['ls2.gamma'],o[mode+'_post_ls2']))
    check(mode+'_original_full_residual',same(o[mode+'_post_ls2']+a['residual1'],o[mode+'_block_output']))
for key in ('mlp_output','post_ls2','block_output'):
    check('original_'+key+'_capture_exact',same(o['original_'+key],a[key]))
    actual=o['merged_'+key].reshape(2,437,1024);expected=a[key].reshape(2,437,1024)
    ap=actual[:,5:].reshape(2,18,24,1024);ep=expected[:,5:].reshape(2,18,24,1024)
    regions={'all_tokens':(actual,expected),'all_patches':(ap,ep),'specials':(actual[:,:5],expected[:,:5]),'merged_patches':(ap[pixelmask],ep[pixelmask]),'unmerged_patches':(ap[~pixelmask],ep[~pixelmask])}
    for name,(av,ev) in regions.items():
        av64,ev64=av.astype(np.float64),ev.astype(np.float64);d=av64-ev64
        v={'elements':av.size,'max_abs':float(abs(d).max()),'rmse':float(np.sqrt(np.mean(d*d))),'relative_l2':float(np.linalg.norm(d)/np.linalg.norm(ev64)), 'bitwise_equal':same(av,ev),'outside_strict_FP32':int((abs(d)>1e-5+1e-4*abs(ev64)).sum())}
        for field,value in v.items():
            recorded=p['errors'][key][name][field]
            check(key+'/'+name+'/'+field,math.isclose(value,recorded,rel_tol=1e-12,abs_tol=1e-14) if isinstance(value,float) else value==recorded)
    check(key+'_unmerged_exact',same(ap[~pixelmask],ep[~pixelmask]))
    check(key+'_special_exact',same(actual[:,:5],expected[:,:5]))
for row in p['rounds']:
    mode=row['mode'];f=PROBE/f"round-{row['index']:02d}-{mode}-block-output.npy"
    check('round_'+str(row['index'])+'_same_output',same(np.load(f,allow_pickle=False),o[mode+'_block_output']))
for mode in ('original','merged'):
    values=[r['end_to_end_seconds'] for r in p['rounds'] if r['mode']==mode]
    check(mode+'_four_measured',len(values)==4)
    check(mode+'_timing_mean',float(np.mean(values))==p['timing_summary'][mode]['mean_seconds'])
check('logical_status',p['status']=='completed_isolated_MLP_approximation_probe' and p['sources_and_inputs_unchanged'] and p['cpu_affinity_restored'] and not p['cleanup_errors'])
result={'status':'passed','checks_passed':len(checks),'checks':checks,'script_sha256':sha(Path(__file__)),'capture_sha256':sha(CAP/'capture.json'),'probe_sha256':sha(PROBE/'probe.json'),'host_execution':'NumPy saved-array arithmetic only; no Torch, real model, predictor, camera or image decoder invoked','quality_accepted':False,'speed_scope':'one isolated MLP branch; predictor overhead separately reported','original_mean_ms':p['timing_summary']['original']['mean_seconds']*1000,'merged_mean_ms':p['timing_summary']['merged']['mean_seconds']*1000,'predictor_ms':p['predictor_forward_seconds']*1000,'plan':p['plan'],'all_token_block_relative_l2':p['errors']['block_output']['all_tokens']['relative_l2']}
out=args.output
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),'receipt_sha256':sha(out)},indent=2))
