#!/usr/bin/env python3
"""RK3588-only original retained group FC2 K4-partition, two real ORT checks, FP16 build.

Keep every original weight value and token. Splitting K changes floating-point
reduction order; the fixed original FP32 ONNX gate must pass before compilation.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
HELPER_SHA = '21d958ef385eaa419e81bef04303f07ffb5b7bb32a4854cd5d17ffba6247680b'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
BASELINE_SHA = '771a9b82007d2feefd65ebbf3f4b0fb27070b79ff7f6d0faab1a8c866da90e43'


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
            and [v.name for v in model.graph.output] == [b['name'] for b in parent['output_bindings']], 'Original group ABI differs')
    nodes, tensors = list(model.graph.node), list(model.graph.initializer)
    node_by_name, tensor_by_name = {n.name:n for n in nodes}, {t.name:t for t in tensors}
    require(len(node_by_name) == len(nodes) and len(tensor_by_name) == len(tensors), 'Unique original names required')
    consumers = {}
    for node in nodes:
        for name in node.input: consumers.setdefault(name, []).append(node.name)
    require(type(chunks) is int and chunks == 4, 'Exactly four K chunks required')
    removed_nodes, removed_tensors, substitutions, added, added_weights, proofs = set(), set(), {}, [], [], []
    occupied = set(node_by_name) | set(tensor_by_name) | {v for n in nodes for v in (*n.input, *n.output)}
    def new_names(names):
        require(not occupied.intersection(names), 'New name collision')
        occupied.update(names)
    raw = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    for pair_index, role in ((i, role) for i in parent['indices'] for role in ('frame', 'global')):
        prefix = f'pair{pair_index}/{role}{pair_index}/'
        hidden, wn, bn = [prefix + suffix for suffix in ('mlp.fc2', 'mlp.fc2.weight_transposed', 'mlp.fc2.bias')]
        mm, bias_add = node_by_name[hidden + '.matmul'], node_by_name[hidden]
        require(mm.op_type == 'MatMul' and list(mm.output) == [mm.name]
                and list(mm.input) == [prefix + 'gelu', wn]
                and bias_add.op_type == 'Add' and list(bias_add.input) == [mm.name, bn]
                and list(bias_add.output) == [hidden], 'Original FC2 MatMul/Add pattern differs')
        require(consumers[wn] == [mm.name] and consumers[bn] == [bias_add.name]
                and consumers[mm.name] == [bias_add.name], 'Original FC2 has another consumer')
        wt, bt = nh.to_array(tensor_by_name[wn]), nh.to_array(tensor_by_name[bn])
        require(wt.shape == (4096, 1024) and bt.shape == (1024,) and wt.dtype == bt.dtype == np.float32
                and np.isfinite(wt).all() and np.isfinite(bt).all(), 'Original parameter shape/dtype/finite differs')
        original_raw = parent['parameter_raw_sha256'][str(pair_index)][role]
        require(raw(wt.T) == original_raw['mlp.fc2.weight'] and raw(bt) == original_raw['mlp.fc2.bias'], 'Original FC2 checkpoint values differ')
        branch_nodes, partials, weights = [], [], []
        stem = prefix + f'fc2_k_split{chunks}/'
        axes, steps = stem + 'slice_axis', stem + 'slice_step'
        new_names((axes, steps))
        added.extend([nh.from_array(np.asarray([-1], np.int64), axes),
                      nh.from_array(np.asarray([1], np.int64), steps)])
        for part in range(chunks):
            start, stop = part * (4096 // chunks), (part + 1) * (4096 // chunks)
            a = np.ascontiguousarray(wt[start:stop, :])
            branch = stem + str(part) + '/'
            weight, starts, ends, activation, matmul = [branch + x for x in ('weight', 'starts', 'ends', 'activation', 'matmul')]
            new_names((weight, starts, ends, activation, matmul))
            t = nh.from_array(a, weight); added.append(t); added_weights.append(t)
            added.extend([nh.from_array(np.asarray([start], np.int64), starts),
                          nh.from_array(np.asarray([stop], np.int64), ends)])
            branch_nodes.extend([h.make_node('Slice', [mm.input[0], starts, ends, axes, steps], [activation], name=activation),
                                 h.make_node('MatMul', [activation, weight], [matmul], name=matmul)])
            partials.append(matmul); weights.append(a)
        summed = partials[0]
        for part in range(1, chunks):
            out = mm.name if part == chunks - 1 else stem + f'sum_{part}'
            if out != mm.name: new_names((out,))
            branch_nodes.append(h.make_node('Add', [summed, partials[part]], [out], name=out))
            summed = out
        require(summed == mm.name and raw(np.concatenate(weights,axis=0).T) == original_raw['mlp.fc2.weight'], 'FC2 K slice reconstruction failed')
        removed_nodes.add(mm.name); removed_tensors.add(wn)
        substitutions[mm.name] = branch_nodes
        proofs.append(dict(pair_index=pair_index, role=role, chunks=chunks, weight_raw_sha256=raw(wt.T), bias_raw_sha256=raw(bt),
                           input_channel_ranges=[[i * (4096//chunks),(i+1) * (4096//chunks)] for i in range(chunks)],
                           activation_slice_axis=-1, original_bias_Add_name=bias_add.name,
                           final_sum_output=mm.name, bias_added_once_after_partial_sum=True))
    count = 2 * len(parent['indices'])
    require(len(proofs) == len(removed_nodes) == len(removed_tensors) == count, 'Every selected frame/global FC2 MatMul required')
    old_nodes = [n.SerializeToString() for n in nodes if n.name not in removed_nodes]
    old_tensors = [t.SerializeToString() for t in tensors if t.name not in removed_tensors]
    old_io = [v.SerializeToString() for v in (*model.graph.input,*model.graph.output)]
    changed = []
    for n in nodes:
        if n.name in substitutions: changed.extend(substitutions[n.name])
        elif n.name not in removed_nodes: changed.append(n)
    del model.graph.node[:]; model.graph.node.extend(changed)
    del model.graph.initializer[:]; model.graph.initializer.extend([t for t in tensors if t.name not in removed_tensors] + added)
    new_names = {n.name for group in substitutions.values() for n in group}
    require(old_nodes == [n.SerializeToString() for n in model.graph.node if n.name not in new_names], 'Other original nodes changed')
    require(old_tensors == [t.SerializeToString() for t in model.graph.initializer if t.name not in {a.name for a in added}], 'Other original weights changed')
    require(old_io == [v.SerializeToString() for v in (*model.graph.input,*model.graph.output)], 'Original two outputs changed')
    require(sum(int(np.prod(t.dims)) for t in added_weights) == count * 4096 * 1024, 'FC2 weight elements duplicated/omitted')
    onnx.checker.check_model(model,full_check=True); onnx.save_model(model,str(destination))
    return dict(replacements=proofs, all_other_original_nodes_protobuf_unchanged=True,
                all_other_original_initializers_protobuf_unchanged=True, original_FC1_nodes_protobuf_unchanged=True,
                original_GELU_nodes_protobuf_unchanged=True, original_FC2_bias_Add_nodes_protobuf_unchanged=True,
                original_LayerScale_and_residual_nodes_protobuf_unchanged=True, original_IO_protobuf_unchanged=True,
                all_original_FC2_parameter_values_preserved=True, FC2_weights_not_duplicated=True,
                FC2_bias_added_once_after_partial_sum=True, FP32_reduction_order_changed=True,
                partial_sum_order='left_associative_in_increasing_K_chunk_order',
                FC2_matmul_replacement_count=count, original_parameter_value_count=36*len(parent['indices']),
                target_global_output_source_pair_index=20 if 20 in parent['indices'] else None,
                original_retained_and_target_output_bindings_protobuf_unchanged=True,
                original_nodes=len(nodes),candidate_nodes=len(changed),original_initializers=len(tensors),candidate_initializers=len(model.graph.initializer))


def check(c, baseline, candidate, cases, output, report):
    import numpy as np
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    bindings = report['output_bindings']
    row=dict(schema='v73_original_VGGT_retained_FC2_K4_check',status='running',chunks=4,
             group_index=report['group_index'],indices=report['indices'],input_shape=[2,53,1024],output_bindings=bindings,
             public_output_sources=report['public_output_sources'],baseline_manifest_sha256=BASELINE_SHA,
             original_parent_manifest_sha256=PARENT_SHA,baseline_graph_sha256=sha(baseline),candidate_onnx_sha256=sha(candidate),
             script_sha256=report['script_sha256'],cases=cases,ORT_version=ort.__version__,optimization='ORT_DISABLE_ALL',
             intra_op_num_threads=1,inter_op_num_threads=1,candidate_accepted=False,cleanup_errors=[],
             FP32_reduction_order_changed=True,partial_sum_order='left_associative_in_increasing_K_chunk_order')
    sessions=[]
    try:
        for path in (baseline,candidate):
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']); sessions.append(session)
            session.disable_fallback()
            require(session.get_providers()==['CPUExecutionProvider'] and [(v.name,v.shape,v.type) for v in session.get_inputs()]==[('input',[2,53,1024],'tensor(float)')]
                    and [(v.name,v.shape,v.type) for v in session.get_outputs()]==[(b['name'],b['shape'],'tensor(float)') for b in bindings], 'ORT ABI/provider differs')
        for case in cases:
            root=Path(case['directory']); x=c.tensor(np,root,case['input'],[2,53,1024])
            case['saved_input']=c.save_tensor(np,output,case['tag']+'-input.npy',x,dict(name='input',shape=[2,53,1024]))
            ys=[]; case['ORT_CPU_seconds']={}
            for label,session in zip(('baseline','candidate'),sessions):
                started=time.perf_counter(); ys.append(session.run([b['name'] for b in bindings],{'input':x}))
                case['ORT_CPU_seconds'][label]=time.perf_counter()-started
            case.update(baseline_outputs=[],candidate_outputs=[],checks=[],
                        original_reference_outputs=[] if case['original_outputs'] is not None else None)
            for i,binding in enumerate(bindings):
                a,b=ys[0][i],ys[1][i]
                for label,value in [('baseline',a),('candidate',b)]:
                    case[label+'_outputs'].append(c.save_tensor(np,output,f"{case['tag']}-{label}-{i}.npy",value,binding))
                item=dict(name=binding['name'],byte_equal_to_baseline=a.tobytes()==b.tobytes(),versus_baseline_FP32_ONNX=c.compare(np,b,a),versus_original_FP32=None)
                if case['original_outputs']:
                    reference=c.tensor(np,root,case['original_outputs'][i],binding['shape'])
                    saved=c.save_tensor(np,output,f"{case['tag']}-original-reference-{i}.npy",reference,binding)
                    saved['original_capture_artifact']=case['original_outputs'][i]
                    case['original_reference_outputs'].append(saved)
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
        report.update(cases=cases,check_receipt_sha256=sha(output/'check.json'),
                      all_candidate_vs_baseline_FP32_ONNX_strict_passed=row.get('all_candidate_vs_baseline_FP32_ONNX_strict_passed',False))


def reuse_checked(c, a, output, report, cases, watched):
    """Reuse the same frozen source/graph/check and pinned saved outputs, no ORT."""
    directory=a.compile_checked.resolve(strict=True)
    receipt=c.pin(directory/'result.json',a.check_result_sha256)
    prior=json.loads(receipt.read_text())
    require(prior['schema']==report['schema'] and prior['status']=='checked_not_quality_accepted'
            and prior['mode']=='check_only' and prior['script_sha256']==sha(__file__)
            and prior['group_index']==a.group_index and prior['baseline_manifest_sha256']==BASELINE_SHA
            and prior['original_parent_manifest_sha256']==PARENT_SHA and prior['helper_sha256']==HELPER_SHA
            and prior['baseline_graph_sha256']==report['baseline_graph_sha256']
            and prior['baseline_group']==report['baseline_group'] and prior['original_pair_parents']==report['original_pair_parents']
            and prior['output_bindings']==report['output_bindings'] and prior['public_output_sources']==report['public_output_sources']
            and prior['configuration']==c.CONFIG and prior['build_options']==c.BUILD
            and prior['operations']==[] and prior['cleanup_errors']==[] and prior['environment_restored'] is True
            and prior['source_and_inputs_unchanged'] is True and prior['candidate_accepted'] is False
            and prior['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] is True,'Completed same-source check-only receipt required')
    check_path=c.pin(directory/'check.json',prior['check_receipt_sha256']);checked=json.loads(check_path.read_text())
    require(checked['schema']=='v73_original_VGGT_retained_FC2_K4_check'
            and checked['status']=='passed_original_FP32_ONNX_equivalence_not_quality_accepted'
            and checked['script_sha256']==sha(__file__) and checked['group_index']==a.group_index
            and checked['baseline_graph_sha256']==report['baseline_graph_sha256']
            and checked['candidate_onnx_sha256']==prior['onnx']['sha256']
            and checked['cases']==prior['cases'] and len(checked['cases'])==len(cases)==2
            and checked['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] is True
            and checked['cleanup_errors']==[] and checked['sessions_dropped'] is True,'Passed fixed FP32 candidate check required')
    watched.extend([receipt,check_path])
    for path,digest in prior['source_and_input_hashes'].items():watched.append(c.pin(path,digest))
    for saved,original in zip(checked['cases'],cases):
        for key in ('tag','directory','capture_manifest_sha256','input','original_outputs'):
            require(saved[key]==original[key],'Checked original input/reference identity differs')
        require(len(saved['checks'])==len(report['output_bindings']) and
                all(x['versus_baseline_FP32_ONNX']['strict_pass'] is True and
                    x['versus_baseline_FP32_ONNX']['outside_tolerance']==0 and
                    x['versus_baseline_FP32_ONNX']['strict_rtol']==1e-4 and
                    x['versus_baseline_FP32_ONNX']['strict_atol']==1e-5 for x in saved['checks']),'Fixed strict output gate differs')
        for item in [saved['saved_input'],*saved['baseline_outputs'],*saved['candidate_outputs'],*(saved['original_reference_outputs'] or [])]:
            source=c.artifact(directory,item);watched.append(source)
            shutil.copyfile(source,output/item['file']);watched.append(c.artifact(output,item))
    candidate_source=c.artifact(directory,prior['onnx']);watched.append(candidate_source)
    candidate=output/'group-fc2-k4.onnx';shutil.copyfile(candidate_source,candidate)
    shutil.copyfile(check_path,output/'check.json')
    report.update(graph_proof=prior['graph_proof'],onnx=metadata(candidate),cases=checked['cases'],
                  check_receipt_sha256=prior['check_receipt_sha256'],check_result_sha256=a.check_result_sha256,
                  compile_from_check_directory=str(directory),all_candidate_vs_baseline_FP32_ONNX_strict_passed=True,
                  ORT_reexecuted_for_compile_checked=False)
    return candidate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group-index',type=int,choices=range(4),required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline-manifest',type=Path,default=BASE/'experiments/vggt-v67-retained-groups-r1/compiled.json')
    p.add_argument('--parent-manifest',type=Path,default=BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--capture-dir',type=Path,default=BASE/'experiments/vggt-v53-capture-r1')
    p.add_argument('--calibration-dir',type=Path,default=BASE/'experiments/vggt-v56-calibration-r1')
    p.add_argument('--reference-helper',type=Path,default=BASE/'src/deploy/rk3588/validation/original_retained_groups/check_compile_retained_groups.py')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--compile-checked',type=Path);p.add_argument('--check-result-sha256');a=p.parse_args()
    if (a.compile_checked is None)!=(a.check_result_sha256 is None) or (a.check_only and a.compile_checked is not None):
        p.error('Supply --compile-checked and --check-result-sha256 together, without --check-only')
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board required')
    require(sha(a.reference_helper)==HELPER_SHA,'Frozen real-case helper changed')
    spec=importlib.util.spec_from_file_location('_original_pair_cases',a.reference_helper); c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    output=a.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    r=dict(schema='v73_original_VGGT_retained_FC2_K4_build',status='running',chunks=4,group_index=a.group_index,
           mode='compile_checked' if a.compile_checked else 'check_only' if a.check_only else 'check_and_compile',
           script_sha256=sha(__file__),helper_sha256=HELPER_SHA,
           baseline_manifest_sha256=BASELINE_SHA,original_parent_manifest_sha256=PARENT_SHA,input_shape=[2,53,1024],
           candidate_accepted=False,quality_accepted=False,realtime_validated=False,configuration=c.CONFIG,build_options=c.BUILD,operations=[],released=None,cleanup_errors=[],
           FP32_reduction_order_changed=True,partial_sum_order='left_associative_in_increasing_K_chunk_order')
    watched=[Path(__file__).resolve(),a.reference_helper.resolve()]; pins={}; sdk=None
    cwd,tmp,started=Path.cwd(),os.environ.get('TMPDIR'),time.perf_counter(); c.write(output/'result.json',r)
    try:
        manifest=c.pin(a.baseline_manifest,BASELINE_SHA); baseline=json.loads(manifest.read_text())
        parent_path=c.pin(a.parent_manifest,PARENT_SHA);parent=json.loads(parent_path.read_text())
        indices,bindings,sources=c.group_spec(a.group_index);row=baseline['blocks'][a.group_index]
        require(baseline['schema']=='v67_original_VGGT_retained_groups_RKNN_manifest' and len(baseline['blocks'])==4
                and baseline['original_parent_manifest_sha256']==PARENT_SHA
                and baseline['checkpoint']==parent['checkpoint']==c.CHECKPOINT
                and parent['schema']=='v55_original_VGGT_24_frame_global_pair_RKNN_manifest'
                and baseline['source_hashes']==parent['source_hashes']
                and baseline['original_target_tokens_global_index']==20
                and baseline['original_target_token_clone_and_norm_must_be_retained'] is True
                and row['role']=='retained_pair_group' and row['index']==a.group_index and row['indices']==indices
                and row['input_name']=='input' and row['input_shape']==[2,53,1024]
                and row['output_bindings']==bindings and row['public_output_sources']==sources
                and row['target_global_output_source_pair_index']==(20 if a.group_index==3 else None)
                and row['parents']==[parent['blocks'][i] for i in indices],'Exact original retained group/target required')
        graph=c.artifact(manifest.parent,row)
        old_check_path=c.artifact(manifest.parent,row['check_receipt']);old_check=json.loads(old_check_path.read_text())
        old_compile_path=c.artifact(manifest.parent,row['compile_receipt']);old_compile=json.loads(old_compile_path.read_text())
        require(old_check['schema']=='v67_original_VGGT_retained_group_check_compile'
                and old_check['status']=='checked_separate_ONNX_equivalence_not_quality_accepted'
                and old_check['all_separate_ONNX_checks_byte_equal'] is True
                and old_compile['status']=='compiled_not_quality_accepted' and old_compile['released'] is True
                and old_compile['release_code'] in (None,0) and old_compile['configuration']==c.CONFIG and old_compile['build_options']==c.BUILD
                and old_compile['check_receipt_sha256']==row['check_receipt']['sha256']
                and all(x['group_index']==a.group_index and x['graph_sha256']==row['sha256']
                        and x['script_sha256']==HELPER_SHA and x['cleanup_errors']==[]
                        and x['source_and_inputs_unchanged'] is True for x in (old_check,old_compile)),
                'Pinned baseline separate-chain check/compile provenance differs')
        watched.extend([manifest,parent_path,graph,old_check_path,old_compile_path])
        cases=c.real_cases(a.capture_dir.resolve(strict=True),a.calibration_dir.resolve(strict=True),parent,watched,a.group_index)
        require(len(cases)==2 and [x['tag'] for x in cases]==['v53','v56-calib00'],'Exactly two distinct real inputs required')
        r.update(indices=indices,baseline_graph_sha256=row['sha256'],baseline_group=row,
                 original_pair_parents=[parent['blocks'][i] for i in indices],output_bindings=bindings,public_output_sources=sources,
                 baseline_check_receipt_sha256=row['check_receipt']['sha256'],baseline_compile_receipt_sha256=row['compile_receipt']['sha256'],
                 retained_consumer_indices=[4,11,17,23],original_target_tokens_global_index=20,
                 original_target_token_clone_and_norm_must_be_retained=True)
        pins={str(path):sha(path) for path in watched};r['source_and_input_hashes']=pins
        if a.compile_checked:
            candidate=reuse_checked(c,a,output,r,cases,watched)
            pins={str(path):sha(path) for path in watched};r['source_and_input_hashes']=pins
        else:
            candidate=output/'group-fc2-k4.onnx';r['graph_proof']=rewrite(graph,candidate,4,row)
            r['onnx']=metadata(candidate);pins[str(candidate)]=r['onnx']['sha256']
            check(c,graph,candidate,cases,output,r)
        pins[str(candidate)]=r['onnx']['sha256']
        pins[str(output/'check.json')]=r['check_receipt_sha256']
        r['status']='checked_not_quality_accepted'; c.write(output/'result.json',r)
        if not a.check_only:
            version=importlib.metadata.version('rknn-toolkit2'); require(version.split('+')[0]=='2.3.2','RKNN2.3.2 required'); r['toolkit_version']=version
            os.chdir(output); os.environ['TMPDIR']=str(output)
            from rknn.api import RKNN
            sdk=RKNN(verbose=True,verbose_file=str(output/'sdk.log')); r['released']=False
            model=output/'group-fc2-k4.rknn'
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
        print(json.dumps(dict(status=r['status'],group_index=a.group_index,chunks=4,result_sha256=sha(output/'result.json'))),flush=True)
    return int(r['status']=='failed')


if __name__=='__main__':
    raise SystemExit(main())
