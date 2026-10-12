#!/usr/bin/env python3
"""RK3588-only original pair00 FC2 N-partition, two real ORT checks, FP16 build."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
HELPER_SHA = '4094e393d0504954a67993c24e82c5534862393930f60db3ec64c2c3e4098b76'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
BINDINGS = [dict(index=0, name='frame_output', shape=[2, 53, 1024], dtype='float32'),
            dict(index=1, name='global_output', shape=[1, 106, 1024], dtype='float32')]


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''): h.update(data)
    return h.hexdigest()


def metadata(path):
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def rewrite(source, destination, chunks, parent):
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper as nh
    model = onnx.load(str(source), load_external_data=False)
    require([v.name for v in model.graph.input] == ['input']
            and [v.name for v in model.graph.output] == [b['name'] for b in BINDINGS], 'Original pair ABI differs')
    nodes, tensors = list(model.graph.node), list(model.graph.initializer)
    node_by_name, tensor_by_name = {n.name:n for n in nodes}, {t.name:t for t in tensors}
    require(len(node_by_name) == len(nodes) and len(tensor_by_name) == len(tensors), 'Unique original names required')
    consumers = {}
    for node in nodes:
        for name in node.input: consumers.setdefault(name, []).append(node.name)
    require(type(chunks) is int and chunks in (2, 4), 'Exactly two/four output-channel chunks required')
    removed_nodes, removed_tensors, substitutions, added, proofs, discarded_values = set(), set(), {}, [], [], set()
    raw = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    for role in ('frame', 'global'):
        prefix = role + '0/'
        hidden, wn, bn = [prefix + suffix for suffix in ('mlp.fc2', 'mlp.fc2.weight_transposed', 'mlp.fc2.bias')]
        mm, bias_add = node_by_name[hidden + '.matmul'], node_by_name[hidden]
        require(mm.op_type == 'MatMul' and list(mm.output) == [mm.name] and mm.input[1] == wn
                and bias_add.op_type == 'Add' and list(bias_add.input) == [mm.name, bn]
                and list(bias_add.output) == [hidden], 'Original FC2 MatMul/Add pattern differs')
        require(consumers[wn] == [mm.name] and consumers[bn] == [bias_add.name]
                and consumers[mm.name] == [bias_add.name], 'Original FC2 has another consumer')
        require(list(mm.input) == [prefix + 'gelu', wn], 'Original FC2 input differs')
        wt, bt = nh.to_array(tensor_by_name[wn]), nh.to_array(tensor_by_name[bn])
        require(wt.shape == (4096, 1024) and bt.shape == (1024,) and wt.dtype == bt.dtype == np.float32
                and np.isfinite(wt).all() and np.isfinite(bt).all(), 'Original parameter shape/dtype/finite differs')
        original_raw = parent['parameter_raw_sha256'][role]
        require(raw(wt.T) == original_raw['mlp.fc2.weight'] and raw(bt) == original_raw['mlp.fc2.bias'], 'Original FC2 checkpoint values differ')
        branch_nodes, output_names, weights, biases = [], [], [], []
        for part in range(chunks):
            start, stop = part * (1024 // chunks), (part + 1) * (1024 // chunks)
            a, b = np.ascontiguousarray(wt[:,start:stop]), np.ascontiguousarray(bt[start:stop])
            stem = prefix + f'fc2_split{chunks}_{part}/'
            weight, bias, matmul, out = [stem + x for x in ('weight', 'bias', 'matmul', 'biased')]
            require(not any(x in tensor_by_name or x in consumers for x in (weight,bias,matmul,out)), 'New name collision')
            added.extend([nh.from_array(a, weight), nh.from_array(b, bias)])
            branch_nodes.extend([h.make_node('MatMul', [mm.input[0],weight], [matmul], name=matmul),
                                 h.make_node('Add', [matmul,bias], [out], name=out)])
            output_names.append(out); weights.append(a); biases.append(b)
        branch_nodes.append(h.make_node('Concat', output_names, [hidden], axis=-1, name=prefix + f'fc2_split{chunks}_concat'))
        require(raw(np.concatenate(weights,axis=1).T) == original_raw['mlp.fc2.weight']
                and raw(np.concatenate(biases)) == original_raw['mlp.fc2.bias'], 'FC2 slice reconstruction failed')
        removed_nodes.update((mm.name,bias_add.name)); removed_tensors.update((wn,bn)); discarded_values.add(mm.name)
        substitutions[mm.name] = branch_nodes
        proofs.append(dict(role=role, chunks=chunks, weight_raw_sha256=raw(wt.T), bias_raw_sha256=raw(bt),
                           output_channel_ranges=[[i * (1024//chunks),(i+1) * (1024//chunks)] for i in range(chunks)]))
    require(len(proofs) == 2 and len(removed_nodes) == len(removed_tensors) == 4, 'Exactly two FC2 edits required')
    old_nodes = [n.SerializeToString() for n in nodes if n.name not in removed_nodes]
    old_tensors = [t.SerializeToString() for t in tensors if t.name not in removed_tensors]
    old_io = [v.SerializeToString() for v in (*model.graph.input,*model.graph.output)]
    changed = []
    for n in nodes:
        if n.name in substitutions: changed.extend(substitutions[n.name])
        elif n.name not in removed_nodes: changed.append(n)
    del model.graph.node[:]; model.graph.node.extend(changed)
    del model.graph.initializer[:]; model.graph.initializer.extend([t for t in tensors if t.name not in removed_tensors] + added)
    info = [v for v in model.graph.value_info if v.name not in discarded_values]
    del model.graph.value_info[:]; model.graph.value_info.extend(info)
    new_names = {n.name for group in substitutions.values() for n in group}
    require(old_nodes == [n.SerializeToString() for n in model.graph.node if n.name not in new_names], 'Other original nodes changed')
    require(old_tensors == [t.SerializeToString() for t in model.graph.initializer if t.name not in {a.name for a in added}], 'Other original weights changed')
    require(old_io == [v.SerializeToString() for v in (*model.graph.input,*model.graph.output)], 'Original two outputs changed')
    require(sum(int(np.prod(t.dims)) for t in added) == 2 * (4096*1024+1024), 'FC2 elements duplicated/omitted')
    onnx.checker.check_model(model,full_check=True); onnx.save_model(model,str(destination))
    return dict(replacements=proofs, all_other_original_nodes_protobuf_unchanged=True,
                all_other_original_initializers_protobuf_unchanged=True, original_GELU_nodes_protobuf_unchanged=True, original_FC1_nodes_protobuf_unchanged=True, original_dot_K_axis_unchanged=True,
                original_IO_protobuf_unchanged=True, all_original_FC2_parameter_values_preserved=True, FC2_weights_not_duplicated=True,
                original_nodes=len(nodes),candidate_nodes=len(changed),original_initializers=len(tensors),candidate_initializers=len(model.graph.initializer))


def check(c, baseline, candidate, cases, output, report):
    import numpy as np
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    row=dict(schema='v76_original_VGGT_pair00_FC2_N_split_ONNX_check',status='running',chunks=report['chunks'],
             original_parent_manifest_sha256=PARENT_SHA,original_pair_onnx_sha256=GRAPH_SHA,candidate_onnx_sha256=sha(candidate),
             script_sha256=report['script_sha256'],cases=cases,ORT_version=ort.__version__,optimization='ORT_DISABLE_ALL',
             intra_op_num_threads=1,inter_op_num_threads=1,candidate_accepted=False,cleanup_errors=[])
    sessions=[]
    try:
        for path in (baseline,candidate):
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']); sessions.append(session)
            session.disable_fallback()
            require(session.get_providers()==['CPUExecutionProvider'] and [(v.name,v.shape,v.type) for v in session.get_inputs()]==[('input',[2,53,1024],'tensor(float)')]
                    and [(v.name,v.shape,v.type) for v in session.get_outputs()]==[(b['name'],b['shape'],'tensor(float)') for b in BINDINGS], 'ORT ABI/provider differs')
        for case in cases:
            root=Path(case['directory']); x=c.tensor(np,root,case['input'],[2,53,1024])
            case['original_outputs']=case['original_outputs'][:2] if case['original_outputs'] else None
            case['saved_input']=c.save_tensor(np,output,case['tag']+'-input.npy',x,dict(name='input',shape=[2,53,1024]))
            ys=[s.run([b['name'] for b in BINDINGS],{'input':x}) for s in sessions]
            case.update(baseline_outputs=[],candidate_outputs=[],checks=[])
            for i,binding in enumerate(BINDINGS):
                a,b=ys[0][i],ys[1][i]
                for label,value in [('baseline',a),('candidate',b)]:
                    case[label+'_outputs'].append(c.save_tensor(np,output,f"{case['tag']}-{label}-{i}.npy",value,binding))
                item=dict(name=binding['name'],byte_equal_to_baseline=a.tobytes()==b.tobytes(),versus_baseline_FP32_ONNX=c.compare(np,b,a),versus_original_FP32=None)
                if case['original_outputs']:
                    reference=c.tensor(np,root,case['original_outputs'][i],binding['shape'])
                    item['versus_original_FP32']={label:c.compare(np,value,reference) for label,value in [('baseline',a),('candidate',b)]}
                case['checks'].append(item)
        row['all_candidate_vs_baseline_FP32_ONNX_strict_passed']=all(v['versus_baseline_FP32_ONNX']['strict_pass'] for case in cases for v in case['checks'])
        row['original_Torch_gate_passed_where_reference_available']=all(v['versus_original_FP32'][label]['strict_pass'] for case in cases for v in case['checks'] if v['versus_original_FP32'] for label in ('baseline','candidate'))
        row['original_Torch_references_cover_all_cases']=False
        require(row['all_candidate_vs_baseline_FP32_ONNX_strict_passed'],'Candidate differs from original FP32 ONNX; compile blocked')
        row['status']='passed_original_FP32_ONNX_equivalence_not_quality_accepted'
    except BaseException as exc:
        row.update(status='failed',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        sessions.clear(); session=None; row['sessions_dropped']=True; c.write(output/'check.json',row)
    report.update(cases=cases,check_receipt_sha256=sha(output/'check.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chunks',type=int,choices=(2,4),required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline-manifest',type=Path,default=BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--capture-dir',type=Path,default=BASE/'experiments/vggt-v53-capture-r1')
    p.add_argument('--calibration-dir',type=Path,default=BASE/'experiments/vggt-v56-calibration-r1')
    p.add_argument('--reference-helper',type=Path,default=BASE/'src/deploy/rk3588/validation/original_pair_group/check_compile_pair_group.py')
    p.add_argument('--check-only',action='store_true'); a=p.parse_args()
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board required')
    require(sha(a.reference_helper)==HELPER_SHA,'Frozen real-case helper changed')
    spec=importlib.util.spec_from_file_location('_original_pair_cases',a.reference_helper); c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    output=a.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    r=dict(schema='v76_original_VGGT_pair00_FC2_N_split_build',status='running',chunks=a.chunks,script_sha256=sha(__file__),helper_sha256=HELPER_SHA,
           original_parent_manifest_sha256=PARENT_SHA,original_pair_onnx_sha256=GRAPH_SHA,output_bindings=BINDINGS,input_shape=[2,53,1024],
           candidate_accepted=False,quality_accepted=False,realtime_validated=False,configuration=c.CONFIG,build_options=c.BUILD,operations=[],released=None,cleanup_errors=[])
    watched=[Path(__file__).resolve(),a.reference_helper.resolve()]; pins={}; sdk=None
    cwd,tmp,started=Path.cwd(),os.environ.get('TMPDIR'),time.perf_counter(); c.write(output/'result.json',r)
    try:
        manifest=c.pin(a.baseline_manifest,PARENT_SHA); parent=json.loads(manifest.read_text()); row=parent['blocks'][0]
        require(parent['schema']=='v55_original_VGGT_24_frame_global_pair_RKNN_manifest' and parent['checkpoint']==c.CHECKPOINT
                and row['role']=='pair' and row['index']==0 and row['sha256']==GRAPH_SHA and row['output_bindings']==BINDINGS,'Exact original pair00 required')
        graph=c.artifact(manifest.parent,row); watched.extend([manifest,graph])
        cases=c.real_cases(a.capture_dir.resolve(strict=True),a.calibration_dir.resolve(strict=True),parent,watched)
        require(len(cases)==2,'Exactly two distinct real inputs required'); pins={str(path):sha(path) for path in watched}
        r['parent']=row; r['source_and_input_hashes']=pins
        candidate=output/f'pair-fc2-split{a.chunks}.onnx'; r['graph_proof']=rewrite(graph,candidate,a.chunks,row)
        r['onnx']=metadata(candidate); pins[str(candidate)]=r['onnx']['sha256']
        check(c,graph,candidate,cases,output,r); r['status']='checked_not_quality_accepted'; c.write(output/'result.json',r)
        if not a.check_only:
            version=importlib.metadata.version('rknn-toolkit2'); require(version.split('+')[0]=='2.3.2','RKNN2.3.2 required'); r['toolkit_version']=version
            os.chdir(output); os.environ['TMPDIR']=str(output)
            from rknn.api import RKNN
            sdk=RKNN(verbose=True,verbose_file=str(output/'sdk.log')); r['released']=False
            model=output/f'pair-fc2-split{a.chunks}.rknn'
            for name,fn in [('config',lambda:sdk.config(**c.CONFIG)),('load_onnx',lambda:sdk.load_onnx(model=str(candidate))),
                            ('build',lambda:sdk.build(**c.BUILD)),('export_rknn',lambda:sdk.export_rknn(str(model)))]:
                operation=dict(name=name,status='running'); r['operations'].append(operation); c.write(output/'result.json',r); t=time.perf_counter()
                code=fn(); operation.update(status='completed',code=code,seconds=time.perf_counter()-t); require(code in (None,0),f'{name} returned {code}')
            r.update(rknn=metadata(model),status='compiled_not_quality_accepted')
    except BaseException as exc:
        r.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if r['operations'] and r['operations'][-1]['status']=='running': r['operations'][-1].update(status='failed',error=r['error'])
    finally:
        if sdk is not None:
            try:
                r['release_code']=sdk.release(); r['released']=r['release_code'] in (None,0); require(r['released'],'SDK release failed')
            except BaseException as exc: r['cleanup_errors'].append(f'release: {type(exc).__name__}: {exc}')
        try:
            os.chdir(cwd)
            if tmp is None: os.environ.pop('TMPDIR',None)
            else: os.environ['TMPDIR']=tmp
            r['environment_restored']=Path.cwd()==cwd and os.environ.get('TMPDIR')==tmp
            r['source_and_input_hashes_after']={path:sha(Path(path)) for path in pins}
            r['source_and_inputs_unchanged']=bool(pins) and pins==r['source_and_input_hashes_after']
            require(r['environment_restored'] and r['source_and_inputs_unchanged'],'Source/input/environment changed')
        except BaseException as exc: r['cleanup_errors'].append(f'restore/identity: {type(exc).__name__}: {exc}')
        if r['cleanup_errors']: r['status']='failed'
        r['seconds']=time.perf_counter()-started; c.write(output/'result.json',r)
        print(json.dumps(dict(status=r['status'],chunks=a.chunks,result_sha256=sha(output/'result.json'))),flush=True)
    return int(r['status']=='failed')


if __name__=='__main__':
    raise SystemExit(main())
