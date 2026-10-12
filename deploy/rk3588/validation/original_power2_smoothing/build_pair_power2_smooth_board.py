#!/usr/bin/env python3
"""Capture original pair Norm outputs, reversibly rescale channels, check FP32."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
HELPER_SHA = '4094e393d0504954a67993c24e82c5534862393930f60db3ec64c2c3e4098b76'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
BOUNDARIES = [(role+'0/'+norm, role+'0/'+linear, [2,53,1024] if role=='frame' else [1,106,1024])
    for role in ('frame','global') for norm,linear in (('norm1','attn.qkv'),('norm2','mlp.fc1'))]
BINDINGS = [dict(index=0,name='frame_output',shape=[2,53,1024],dtype='float32'),
            dict(index=1,name='global_output',shape=[1,106,1024],dtype='float32')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def info(path):
    return dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))


def raw(np,x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def rewrite(np,onnx,model,maxima,parent):
    from onnx import numpy_helper as nh
    candidate=copy.deepcopy(model)
    nodes={n.name:n for n in candidate.graph.node}
    tensors={t.name:t for t in candidate.graph.initializer}
    consumers={}
    for node in candidate.graph.node:
        for x in node.input:consumers.setdefault(x,[]).append(node.name)
    replacements={};proofs=[]
    for norm,linear,shape in BOUNDARIES:
        role=norm.split('0/')[0];shortnorm=norm.split('/',1)[1];shortlinear=linear.split('/',1)[1]
        gamma,beta,wn=norm+'.weight',norm+'.bias',linear+'.weight_transposed'
        assert nodes[norm].op_type=='LayerNormalization' and list(nodes[norm].input[1:])==[gamma,beta]
        assert nodes[linear+'.matmul'].op_type=='MatMul'
        assert list(nodes[linear+'.matmul'].input)==[norm,wn] and consumers[norm]==[linear+'.matmul']
        assert consumers[gamma]==consumers[beta]==[norm] and consumers[wn]==[linear+'.matmul']
        g,b,w=[nh.to_array(tensors[name]) for name in (gamma,beta,wn)]
        assert g.shape==b.shape==(1024,) and w.shape==(1024,3072 if shortnorm=='norm1' else 4096)
        originals=parent['parameter_raw_sha256'][role]
        assert raw(np,g)==originals[shortnorm+'.weight'] and raw(np,b)==originals[shortnorm+'.bias']
        assert raw(np,w.T)==originals[shortlinear+'.weight']
        weight_max=np.max(np.abs(w.astype(np.float64)),axis=1)
        exponent=np.rint(.5*np.log2(np.maximum(maxima[norm],1e-5))-.5*np.log2(np.maximum(weight_max,1e-5)))
        assert np.isfinite(exponent).all() and np.max(np.abs(exponent))<100
        scales=np.exp2(exponent).astype(np.float32)
        values=[g/scales,b/scales,w*scales[:,None]]
        reverse=[values[0]*scales,values[1]*scales,values[2]/scales[:,None]]
        assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in values)
        assert all(raw(np,x)==raw(np,y) for x,y in zip((g,b,w),reverse)), 'Scaling not raw-byte reversible'
        for name,value in zip((gamma,beta,wn),values):replacements[name]=nh.from_array(value,name)
        unique,counts=np.unique(exponent.astype(int),return_counts=True)
        proofs.append(dict(norm=norm,linear=linear,alpha=.5,scale_raw_sha256=raw(np,scales),
            exponent_counts={str(k):int(v) for k,v in zip(unique,counts)},inverse_bytes_reconstructed=True,
            before_raw_sha256=[raw(np,v) for v in (g,b,w)],after_raw_sha256=[raw(np,v) for v in values]))
    assert len(replacements)==12
    for i,t in enumerate(candidate.graph.initializer):
        if t.name in replacements:candidate.graph.initializer[i].CopyFrom(replacements[t.name])
    assert [n.SerializeToString() for n in model.graph.node]==[n.SerializeToString() for n in candidate.graph.node]
    assert [t.SerializeToString() for t in model.graph.initializer if t.name not in replacements]==[
        t.SerializeToString() for t in candidate.graph.initializer if t.name not in replacements]
    assert [x.SerializeToString() for x in (*model.graph.input,*model.graph.output)]==[
        x.SerializeToString() for x in (*candidate.graph.input,*candidate.graph.output)]
    onnx.checker.check_model(candidate,full_check=True)
    return candidate,dict(boundaries=proofs,all_nodes_protobuf_unchanged=True,other_initializers_protobuf_unchanged=True,
        original_IO_protobuf_unchanged=True,changed_initializers=12,inverse_bytes_reconstructed=True,
        all_original_parameter_values_literally_unchanged=False,checkpoint_file_modified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
    helper=BASE/'src/deploy/rk3588/validation/original_pair_group/check_compile_pair_group.py'
    assert sha(helper)==HELPER_SHA
    spec=importlib.util.spec_from_file_location('_original_pair_checks',helper);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    root=a.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(schema='v78_original_pair_power2_smooth_FP32_check',status='running',script_sha256=sha(__file__),
        original_model_only=True,original_pair_onnx_sha256=GRAPH_SHA,parent_manifest_sha256=PARENT_SHA,
        alpha=.5,calibration_host_execution=False,NPU_or_SDK_execution=False,candidate_accepted=False,cleanup_errors=[])
    sessions=[];pins={};begin=time.monotonic()
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        manifest=c.pin(BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json',PARENT_SHA)
        parent=json.loads(manifest.read_text());row=parent['blocks'][0]
        assert parent['checkpoint']==c.CHECKPOINT and row['sha256']==GRAPH_SHA and row['output_bindings']==BINDINGS
        graph=c.artifact(manifest.parent,row);watched=[Path(__file__).resolve(),helper,manifest,graph]
        calibration=BASE/'experiments/vggt-v56-calibration-r1'
        cases=c.real_cases(BASE/'experiments/vggt-v53-capture-r1',calibration,parent,watched)
        capture=json.loads((calibration/'capture.json').read_text());group=capture['groups']['aggregator'][0]
        assert len(group['samples'])==8 and group['index']==0
        dataset=c.artifact(calibration,group['dataset']);watched.append(dataset)
        cal_paths=[c.artifact(calibration,x['artifact']) for x in group['samples']]
        assert dataset.read_text().splitlines()==[str(x) for x in cal_paths];watched.extend(cal_paths)
        pins={str(path):sha(path) for path in watched}
        calibration_images={image['sha256'] for sample in capture['samples'] for image in sample['images']}
        report['v53_image_SHA_overlap_with_calibration']=sorted({im['sha256'] for im in cases[0]['source_images']} & calibration_images)
        model=onnx.load(str(graph));aug=copy.deepcopy(model)
        for norm,_,shape in BOUNDARIES:aug.graph.output.append(onnx.helper.make_tensor_value_info(norm,onnx.TensorProto.FLOAT,shape))
        onnx.checker.check_model(aug,full_check=True);augpath=root/'original-norm-capture.onnx';onnx.save(aug,str(augpath))
        report['capture_graph']=info(augpath)
        opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        def session(path):
            s=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider']);s.disable_fallback()
            assert s.get_providers()==['CPUExecutionProvider'];sessions.append(s);return s
        original=session(graph);probe=session(augpath)
        maxima={norm:np.zeros(1024,dtype=np.float64) for norm,_,_ in BOUNDARIES};report['calibration']=[]
        for item in group['samples']:
            x=c.tensor(np,calibration,item['artifact'],[2,53,1024]);ys=probe.run(None,{'input':x})
            assert len(ys)==6
            saved={};records=[]
            for (norm,_,shape),y in zip(BOUNDARIES,ys[2:]):
                assert y.dtype==np.float32 and list(y.shape)==shape and np.isfinite(y).all()
                maxima[norm]=np.maximum(maxima[norm],np.max(np.abs(y.astype(np.float64)),axis=(0,1)))
                saved[norm]=y;records.append(dict(name=norm,shape=shape,raw_sha256=raw(np,y)))
            path=root/(item['sample_id']+'-norms.npz');np.savez(path,**saved)
            report['calibration'].append(dict(sample_id=item['sample_id'],input=item['artifact'],norms=records,artifact=info(path)))
        candidate,report['graph_proof']=rewrite(np,onnx,model,maxima,row)
        candidatepath=root/'pair-power2-smooth.onnx';onnx.save(candidate,str(candidatepath));report['onnx']=info(candidatepath)
        np.savez(root/'norm-absmax.npz',**maxima);report['norm_absmax']=info(root/'norm-absmax.npz')
        smooth=session(candidatepath)
        for case in cases:
            x=c.tensor(np,Path(case['directory']),case['input'],[2,53,1024])
            case['original_outputs']=case['original_outputs'][:2] if case['original_outputs'] else None
            case['saved_input']=c.save_tensor(np,root,case['tag']+'-input.npy',x,dict(name='input',shape=[2,53,1024]))
            outputs=[s.run([b['name'] for b in BINDINGS],{'input':x}) for s in (original,smooth)]
            case.update(baseline_outputs=[],candidate_outputs=[],checks=[])
            for i,b in enumerate(BINDINGS):
                u,v=outputs[0][i],outputs[1][i]
                for label,y in [('baseline',u),('candidate',v)]:
                    case[label+'_outputs'].append(c.save_tensor(np,root,f'{case["tag"]}-{label}-{i}.npy',y,b))
                check=dict(name=b['name'],byte_equal_to_baseline=u.tobytes()==v.tobytes(),versus_baseline_FP32_ONNX=c.compare(np,v,u),versus_original_FP32=None)
                if case['original_outputs']:
                    ref=c.tensor(np,Path(case['directory']),case['original_outputs'][i],b['shape'])
                    check['versus_original_FP32']={label:c.compare(np,y,ref) for label,y in [('baseline',u),('candidate',v)]}
                case['checks'].append(check)
        passed=all(x['versus_baseline_FP32_ONNX']['strict_pass'] for case in cases for x in case['checks'])
        check=dict(schema='v78_original_pair_power2_smooth_ONNX_check',script_sha256=sha(__file__),
            original_pair_onnx_sha256=GRAPH_SHA,candidate_onnx_sha256=sha(candidatepath),cases=cases,
            all_candidate_vs_baseline_FP32_ONNX_strict_passed=passed,
            status='passed_original_FP32_ONNX_equivalence_not_quality_accepted' if passed else 'failed_original_FP32_gate',
            ORT_version=ort.__version__,optimization='ORT_DISABLE_ALL',candidate_accepted=False)
        c.write(root/'check.json',check);report.update(cases=cases,check_receipt_sha256=sha(root/'check.json'))
        assert passed,'Original FP32 gate failed; do not compile'
        report['status']='checked_original_FP32_equivalence_not_quality_accepted'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        sessions.clear()
        report.update(seconds=time.monotonic()-begin,source_and_inputs_unchanged=True)
        for path,digest in pins.items():
            try:assert sha(path)==digest,path
            except BaseException as exc:
                report['source_and_inputs_unchanged']=False;report['cleanup_errors'].append(str(exc))
        if report['cleanup_errors']:report['status']='failed_identity'
        c.write(root/'result.json',report)
        print(json.dumps({k:report.get(k) for k in ('status','seconds','error','cleanup_errors')}),flush=True)


if __name__=='__main__':main()
