#!/usr/bin/env python3
"""Board-only original native518 pair basis export and unchanged FP32 gates.

A: pinned B1 frame0 followed by pinned global0 FC2K4. F: identity-basis affine
and LayerScale folds. B: fixed mean-preserving residual basis, persistent across
the two blocks, with canonical exits. No checkpoint deserialization or RKNN.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

FRAME = [2,1041,1024]
GLOBAL = [1,2082,1024]
CHECKPOINT = dict(bytes=5026874952,sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
PINS = dict(original_export='b922ca37e4052800c44520ad7826be97ff463c9d6298e695b70f3d1a7d9c115e',
    frame_export='54cdac3b68da2f07d1fb53fcda94d4a4f3ac3b0994ebdc0320b7b1fac20ac447',
    global_export='7ad10c71635233e2056dade6f0b6c769e0ae0c1dfdf192514f1be18e3afdc9ff',
    weights_capture='4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67',
    native_capture='4ebbc4adaaed5e3c3cf8558b4947ec3fc090d56e82f8c61c9509951eb34dd35d',
    algebra_receipt='08036d7c9702c4c854752ab98c476af298b2736fae45817d22a71394df3a31e5',
    algebra_source='0130df51f4aaa5aa7d43d2bbae470aa8b3525a900bf1417e16d3fb6d2c63acbe',
    frame_onnx='683b4ec52a6d4692fa777b2180a143de347b4765edded313717946c5a0a450f9',
    global_onnx='c44094793c3b590c6066cbdd634cf0c4d8bd5b75987c8b491db7b17b2201f55d')
FIXED_GATE = dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5)
LINEARS = ('attn.qkv','attn.proj','mlp.fc1','mlp.fc2')


def require(ok,message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def raw(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def metadata(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),tensor_bytes=value.nbytes,
        raw_sha256=raw(value),finite=bool(__import__('numpy').isfinite(value).all()))


def rotate64(value,axis,np):
    """R tensor-power5 along one1024 axis; float64 offline, no dense R product."""
    require(value.shape[axis] == 1024,'Original residual axis must be1024')
    moved = np.moveaxis(np.asarray(value,dtype=np.float64),axis,-1).copy(order='C')
    stride = 1
    while stride < 1024:
        blocks = moved.reshape(*moved.shape[:-1],1024//(4*stride),4,stride)
        previous = blocks.copy()
        half_sum = previous.sum(axis=-2,dtype=np.float64)*.5
        blocks[...] = half_sum[...,None,:]-previous
        stride *= 4
    return np.ascontiguousarray(np.moveaxis(moved,-1,axis))


def learned_state(model,role,parameters,np,onnx):
    arrays = {t.name:onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    require(len(arrays) == len(model.graph.initializer),'Duplicate original initializer')
    state = {}
    for name,row in parameters.items():
        stored = name+'_transposed' if name.endswith('.weight') and name.rsplit('.',1)[0] in LINEARS else name
        if name == 'mlp.fc2.weight' and role == 'global':
            pieces = [arrays[stored+'.part'+str(i)] for i in range(4)]
            require(all(p.shape == (1024,1024) and p.dtype == np.float32 for p in pieces),'Original FC2 K4 pieces differ')
            value = np.concatenate(pieces,axis=0).T
        else:
            value = arrays[stored]
            if stored.endswith('_transposed'):value = value.T
        value = np.ascontiguousarray(value)
        require(value.dtype == np.float32 and list(value.shape) == row['shape'] and
            raw(value) == row['raw_sha256'] and np.isfinite(value).all(),'Original learned initializer differs: '+role+'/'+name)
        state[name] = value
    require(len(state) == 18,'All18 original learned values required per block')
    return state


def fold(model,state,role,basis,np,onnx):
    """Original NK state -> derived row-linear storage, no nonlinear movement."""
    updated = copy.deepcopy(model)
    replacements,derived = {},{}
    for norm,linear in (('norm1','attn.qkv'),('norm2','mlp.fc1')):
        w = state[linear+'.weight'].T.astype(np.float64)
        scaled = state[norm+'.weight'].astype(np.float64)[:,None]*w
        result = rotate64(scaled,0,np) if basis else scaled
        replacements[linear+'.weight_transposed'] = np.ascontiguousarray(result,dtype=np.float32)
        replacements[linear+'.bias'] = np.ascontiguousarray(
            state[norm+'.bias'].astype(np.float64)@w+state[linear+'.bias'].astype(np.float64),dtype=np.float32)
    for linear,scale in (('attn.proj','ls1.gamma'),('mlp.fc2','ls2.gamma')):
        scaled = state[linear+'.weight'].T.astype(np.float64)*state[scale].astype(np.float64)[None,:]
        result = rotate64(scaled,1,np) if basis else scaled
        bias = state[linear+'.bias'].astype(np.float64)*state[scale].astype(np.float64)
        bias = rotate64(bias,0,np) if basis else bias
        replacements[linear+'.weight_transposed'] = np.ascontiguousarray(result,dtype=np.float32)
        replacements[linear+'.bias'] = np.ascontiguousarray(bias,dtype=np.float32)
    full_fc2 = replacements.pop('mlp.fc2.weight_transposed') if role == 'global' else None
    if full_fc2 is not None:
        for i in range(4):replacements['mlp.fc2.weight_transposed.part'+str(i)] = np.ascontiguousarray(full_fc2[i*1024:(i+1)*1024])
        require(np.array_equal(np.concatenate([replacements['mlp.fc2.weight_transposed.part'+str(i)] for i in range(4)],axis=0),full_fc2),
            'Derived original FC2 K4 reconstruction failed')
    expected = 11 if role == 'global' else 8
    require(len(replacements) == expected,'Unexpected derived linear arity')
    old_initializers = {t.name:t.SerializeToString() for t in model.graph.initializer}
    for tensor in updated.graph.initializer:
        if tensor.name in replacements:
            value = replacements[tensor.name]
            require(np.isfinite(value).all(),'Nonfinite derived constant')
            tensor.CopyFrom(onnx.numpy_helper.from_array(value,tensor.name))
            derived[tensor.name] = metadata(value)
    require(set(derived) == set(replacements),'Derived initializer missing from parent')
    unit,zero = '__v126_bare_norm_unit','__v126_bare_norm_zero'
    require(unit not in old_initializers and zero not in old_initializers,'Basis name collision')
    updated.graph.initializer.extend([onnx.numpy_helper.from_array(np.ones(1024,np.float32),unit),
        onnx.numpy_helper.from_array(np.zeros(1024,np.float32),zero)])
    changed_nodes,affines,scales = [],0,0
    for node in updated.graph.node:
        if node.op_type == 'LayerNormalization' and list(node.input[1:]) in (
                ['norm1.weight','norm1.bias'],['norm2.weight','norm2.bias']):
            attrs = {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            require(attrs['axis'] == -1 and abs(attrs['epsilon']-1e-5)<1e-11,'Original LayerNorm axis/epsilon differs')
            node.input[1:] = [unit,zero]
            affines += 1;changed_nodes.append(node.name)
        if node.op_type == 'Mul' and len(node.input) == 2 and node.input[1] in ('ls1.gamma','ls2.gamma'):
            require(not node.attribute,'Unexpected original LayerScale attributes')
            node.op_type = 'Identity';del node.input[1:]
            scales += 1;changed_nodes.append(node.name)
    branches = 2 if role == 'frame' else 1
    require(affines == scales == 2*branches,'Complete original norm/LayerScale fold sites required')
    before = {n.name:n.SerializeToString() for n in model.graph.node}
    require(len(before) == len(model.graph.node),'Duplicate original node')
    require(all(n.SerializeToString() == before[n.name] for n in updated.graph.node if n.name not in changed_nodes),
        'Nonlinear, RoPE, residual or original K4 sum node changed')
    require(all(t.SerializeToString() == old_initializers[t.name] for t in updated.graph.initializer
        if t.name in old_initializers and t.name not in replacements),'Unrelated original initializer changed')
    # Prune only unused tiny affine/LayerScale vectors, not duplicate full weights.
    used = {v for n in updated.graph.node for v in n.input}|{v.name for v in updated.graph.output}
    removed = [t.name for t in updated.graph.initializer if t.name not in used]
    kept = [t for t in updated.graph.initializer if t.name in used]
    del updated.graph.initializer[:];updated.graph.initializer.extend(kept)
    del updated.graph.value_info[:]
    return updated,dict(role=role,basis='mean_preserving_R' if basis else 'identity_R',
        original_parameter_metadata=copy.deepcopy({k:metadata(v) for k,v in state.items()}),
        derived_initializer_metadata=derived,modified_nodes=changed_nodes,removed_unused_initializers=removed,
        original_K4_slices_and_sum_nodes_preserved=role == 'global',
        all_other_nodes_and_used_initializers_protobuf_unchanged=True,
        original_learned_values_recovered_and_verified=18,original_gamma_beta_and_LayerScale_folded=True,
        original_initializer_byte_preservation_claimed=False,finite_precision_equivalence_claimed=False)


def compose(frame,global_model,variant,np,onnx):
    H,T = onnx.helper,onnx.TensorProto
    require(frame.ir_version == global_model.ir_version and [v.SerializeToString() for v in frame.opset_import] ==
        [v.SerializeToString() for v in global_model.opset_import],'Original parent IR/opsets differ')
    nodes,initializers = [],[]
    basis = variant == 'B'
    frame_out,global_out = ('__frame_basis','__global_basis') if basis else ('frame_output','global_output')
    frame_in = '__input_basis' if basis else 'input'
    if basis:
        r = np.ascontiguousarray(rotate64(np.eye(1024,dtype=np.float64),1,np),dtype=np.float32)
        require(np.array_equal(r,r.T) and np.array_equal(r.sum(axis=1),np.ones(1024,np.float32))
            and set(np.unique(r)) == {-.03125,.03125},'Fixed mean-preserving basis construction differs')
        initializers.append(onnx.numpy_helper.from_array(r,'__v126_R'))
        nodes.append(H.make_node('MatMul',['input','__v126_R'],[frame_in],name='__v126_initial_basis'))
    def append(model,prefix,input_name,output_name):
        require(len(model.graph.input) == len(model.graph.output) == 1,'Original single boundary required')
        old_in,old_out = model.graph.input[0].name,model.graph.output[0].name
        mapping = {old_in:input_name,old_out:output_name}
        rename = lambda name: mapping.get(name,prefix+name) if name else name
        for tensor in model.graph.initializer:
            value = copy.deepcopy(tensor);value.name = rename(tensor.name);initializers.append(value)
        for original in model.graph.node:
            node = copy.deepcopy(original);node.name = prefix+node.name
            node.input[:] = [rename(v) for v in original.input];node.output[:] = [rename(v) for v in original.output]
            nodes.append(node)
    append(frame,'frame0/',frame_in,frame_out)
    initializers.append(onnx.numpy_helper.from_array(np.asarray(GLOBAL,np.int64),'__v126_global_shape'))
    nodes.append(H.make_node('Reshape',[frame_out,'__v126_global_shape'],['__global_input'],name='__v126_original_frame_global_view'))
    append(global_model,'global0/','__global_input',global_out)
    if basis:
        for src,out in ((frame_out,'frame_output'),(global_out,'global_output')):
            nodes.append(H.make_node('MatMul',[src,'__v126_R'],[out],name='__v126_restore_'+out))
    graph = H.make_graph(nodes,'original_native518_frame0_global0_basis_'+variant,
        [H.make_tensor_value_info('input',T.FLOAT,FRAME)],
        [H.make_tensor_value_info('frame_output',T.FLOAT,FRAME),H.make_tensor_value_info('global_output',T.FLOAT,GLOBAL)],initializers)
    model = H.make_model(graph,opset_imports=[copy.deepcopy(v) for v in frame.opset_import]);model.ir_version = frame.ir_version
    model.producer_name = 'Edge-VGGT-SLAM/original-residual-basis-board-v126'
    model.doc_string = 'Original B1 frame0 then original FC2K4 global0. All36 original learned values retained or algebraically derived; original QK/RoPE/Erf. Canonical endpoints. FP32 accuracy and hardware use require actual gates.'
    onnx.checker.check_model(model,full_check=True)
    model = onnx.shape_inference.infer_shapes(model,strict_mode=True,check_type=True)
    onnx.checker.check_model(model,full_check=True)
    return model


def comparison(actual,reference,np):
    require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32
        and np.isfinite(actual).all() and np.isfinite(reference).all(),'Finite complete FP32 endpoints required')
    a,b = actual.astype(np.float64),reference.astype(np.float64)
    delta = a-b;maximum = float(np.abs(delta).max());denominator = float(np.linalg.norm(b))
    require(denominator > 0,'Zero original reference norm')
    l2 = float(np.linalg.norm(delta)/denominator)
    count = int(np.count_nonzero(np.abs(delta)>FIXED_GATE['atol']+FIXED_GATE['rtol']*np.abs(b)))
    strict = count == 0
    numeric = l2 <= FIXED_GATE['relative_l2_max'] and maximum <= FIXED_GATE['max_abs_max']
    return dict(relative_l2=l2,max_abs=maximum,outside_count=count,strict_pass=strict,
        fixed_numeric_gate_passed=numeric,fixed_gate_passed=numeric,
        byte_equal=actual.tobytes()==reference.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['export','check'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--export-receipt',type=Path);p.add_argument('--export-sha256')
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588S CPU-only execution required')
    require((a.mode == 'export') or (a.export_receipt is not None and a.export_sha256),'Check requires actual export receipt andSHA')
    out = a.output_dir.resolve();require(str(out).startswith(('/dev/shm/','/tmp/')),'Fresh private scratch output required')
    out.mkdir(parents=True,exist_ok=False)
    began = time.perf_counter();env,affinity = dict(os.environ),set(os.sched_getaffinity(0))
    watched = {};session = None
    report = dict(schema='v126_original_native518_basis_pair_'+a.mode,mode=a.mode,status='running',
        script_sha256=sha(__file__),config_sha256=a.config_sha256,checkpoint=CHECKPOINT,
        original_model_only=True,checkpoint_deserialized=False,actual_execution_on_board_only=True,
        model_packages_Torch_RKNN_GPU_or_host_NN_executed=False,original36_parameter_lineage_required=True,
        fixed_gate=FIXED_GATE,fixed_gate_definition='relative_l2 <= 1e-5 and max_abs <= 1e-3, unchanged original R98 B1 gate',
        strict_elementwise_allclose_is_independent_diagnostic=True,
        models=[],checks=[],cleanup_errors=[],candidate_accepted=False,
        full_model_quality_performance_realtime_or_8GB_accepted=False)
    def progress(stage):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-began,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temp = out/'result.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(out/'result.json')
        print(json.dumps(dict(stage=stage,status=report['status'])),flush=True)
    def pin(path,expected,size=None):
        file = Path(path).resolve(strict=True)
        require(len(expected) == 64 and sha(file) == expected and (size is None or file.stat().st_size == size),'Pinned input differs: '+str(file))
        watched[file] = expected;return file
    def artifact(row):return pin(row['file'],row['sha256'],row.get('bytes'))
    def save(name,value,np):
        file = out/name
        with file.open('xb') as stream:np.save(stream,value,allow_pickle=False)
        return dict(file=name,bytes=file.stat().st_size,sha256=sha(file),**metadata(value))
    try:
        require({4,5,6,7} <= affinity,'Original big-core CPU affinity unavailable')
        os.sched_setaffinity(0,{4,5,6,7});os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        conf = json.loads(pin(a.config,a.config_sha256).read_text())
        require(conf['schema'] == 'v126_original_native518_basis_pair_board_inputs','Input config schema differs')
        for key,expected in PINS.items():require(conf[key]['sha256'] == expected,'Frozen parent identity differs: '+key)
        receipts = {key:json.loads(artifact(conf[key]).read_text()) for key in
            ('original_export','frame_export','global_export','weights_capture','native_capture','algebra_receipt')}
        artifact(conf['algebra_source'])
        base,fr,gr,weights,capture,proof = (receipts[key] for key in
            ('original_export','frame_export','global_export','weights_capture','native_capture','algebra_receipt'))
        require(base['status'] == 'exported_original_native518_blocks_not_FP32_accepted' and base['source_and_inputs_unchanged']
            and base['checkpoint'] == weights['checkpoint'] == capture['checkpoint'] == CHECKPOINT,'Original checkpoint/artifact lineage differs')
        require(fr['status'] == 'exported_not_FP32_or_RKNN_accepted' and fr['source_and_inputs_unchanged']
            and fr['baseline_export_sha256'] == PINS['original_export'] and fr['original_weights_capture_sha256'] == PINS['weights_capture']
            and fr['capture_sha256'] == PINS['native_capture'] and fr['original_parameter_count'] == 18,'Original B1 provenance differs')
        require(gr['status'] == 'static_partition_export_not_quality_accepted' and gr['parts'] == 4
            and gr['source_onnx_sha256'] == next(r for r in base['blocks'] if r['role'] == 'global')['onnx']['sha256']
            and gr['weights_reconstructed_bitwise_equal'] and gr['original_bias_added_once'] and gr['all_other_nodes_and_initializers_preserved'],
            'Original FC2 K4 provenance differs')
        require(capture['completed'] and capture['status'] == 'completed_original_native518_block_references' and
            capture['cleanup_errors'] == [] and capture['image_shape'] == [2,3,392,518] and
            proof['status'] == 'passed_generated_scalar_algebra_only' and proof['script_sha256'] == PINS['algebra_source'],
            'Native input/reference or generated algebra lineage differs')
        source_root = Path(conf['source_root']).resolve(strict=True)
        require(base['source_hashes'] == capture['source_hashes'] == fr['source_hashes'],'Original source authorities differ')
        for relative,expected in capture['source_hashes'].items():pin(source_root/relative,expected)
        require(fr['candidate']['sha256'] == conf['frame_onnx']['sha256'] and gr['output']['sha256'] == conf['global_onnx']['sha256'],
            'Chosen original parent graph differs')
        parent_paths = {role:artifact(conf[role+'_onnx']) for role in ('frame','global')}
        parameters = {role:next(r for r in weights['blocks'] if r['role'] == role and r['index'] == 0)['parameters'] for role in ('frame','global')}
        for role in parameters:
            require(parameters[role] == next(r for r in base['blocks'] if r['role'] == role)['parameters'],'Original36parameter metadata differs')
        require(len(conf['inputs']) == 2 and {r['id'] for r in conf['inputs']} == {'cpu_prefix','npu_prefix'},'Two actual CPU/NPU boundary cases required')
        for case in conf['inputs']:
            require(case['source_pair_indices'] == [32,40] and case['input']['shape'] == FRAME and case['input']['dtype'] == 'float32','Original real input dimensions differ')
            artifact(case['input'])
            for image in case['source_images']:artifact(image)
            if 'lineage_receipt' in case:artifact(case['lineage_receipt'])
        report.update(input_cases=conf['inputs'],input_case_scope=conf['scope'],source_hashes=capture['source_hashes'],
            parent_graphs={'frame':conf['frame_onnx'],'global':conf['global_onnx']},parent_receipts={k:conf[k] for k in receipts},
            original_parameters=parameters,source_pair_indices=[32,40],image_shape=[2,3,392,518],
            checkpoint_and_original_source_lineage_verified=True,source_image_files_decoded=False)
        import numpy as np
        if a.mode == 'export':
            import onnx
            parents = {role:onnx.load_model(str(path),load_external_data=False) for role,path in parent_paths.items()}
            require(all(all(t.data_location != onnx.TensorProto.EXTERNAL for t in model.graph.initializer) for model in parents.values()),'Self-contained original graph required')
            states = {role:learned_state(model,role,parameters[role],np,onnx) for role,model in parents.items()}
            report['all36_original_learned_values_recovered_and_verified'] = True
            for variant in ('A','F','B'):
                progress('export_'+variant)
                folded,details = {},[]
                for role in ('frame','global'):
                    if variant == 'A':folded[role] = parents[role]
                    else:
                        folded[role],detail = fold(parents[role],states[role],role,variant == 'B',np,onnx);details.append(detail)
                model = compose(folded['frame'],folded['global'],variant,np,onnx)
                file = out/('pair-'+variant+'.onnx');onnx.save_model(model,str(file))
                row = dict(key=variant,onnx=dict(file=file.name,bytes=file.stat().st_size,sha256=sha(file)),
                    input_bindings=[dict(index=0,name='input',shape=FRAME,dtype='float32')],
                    output_bindings=[dict(index=0,name='frame_output',shape=FRAME,dtype='float32'),dict(index=1,name='global_output',shape=GLOBAL,dtype='float32')],
                    nodes=len(model.graph.node),initializers=len(model.graph.initializer),fold_proofs=details,
                    basis_persistent_frame_to_global=variant == 'B',canonical_outputs_restored=True,
                    original_K4_partition_and_sum_node_order_preserved=True,original36_learned_values_verified=True)
                report['models'].append(row)
                del model,folded,details;gc.collect()
            predicted = sum(r['onnx']['bytes'] for r in report['models'])+2*3*2*(2082*1024*4+128)
            report['predicted_export_plus12_output_bytes'] = predicted
            require(predicted <= conf['planned_output_byte_budget'],'Planned artifact byte budget exceeded')
            report.update(status='exported_original_basis_pair_not_numerically_accepted',
                NN_ORt_or_RKNN_execution=False,FP32_gate_passed=False,offline_fold_arithmetic='float64 factorizedR thenfloat32 constants',
                changed_fp_rounding_and_internal_precision_unvalidated=True,large_original_weights_duplicated=False,
                basis_io='One initialR andtwo canonical inverseR MatMuls; no between-block inverse or per-blockR',
                external_IO_basis_matmuls_are_not_a_deployment_speed_claim=True)
            del parents,states;gc.collect()
        else:
            ep = pin(a.export_receipt,a.export_sha256);exported = json.loads(ep.read_text())
            require(exported['schema'] == 'v126_original_native518_basis_pair_export' and
                exported['status'] == 'exported_original_basis_pair_not_numerically_accepted' and
                exported['config_sha256'] == a.config_sha256 and exported['script_sha256'] == report['script_sha256'] and
                exported['source_and_inputs_unchanged'] and not exported['cleanup_errors'] and
                [r['key'] for r in exported['models']] == ['A','F','B'] and exported['all36_original_learned_values_recovered_and_verified'],
                'Actual clean export receipt required before FP32 gate')
            report.update(export_receipt=dict(file=str(ep),sha256=a.export_sha256),models=exported['models'])
            graphs = {r['key']:pin(ep.parent/r['onnx']['file'],r['onnx']['sha256'],r['onnx']['bytes']) for r in exported['models']}
            import onnxruntime as ort
            options = ort.SessionOptions();options.intra_op_num_threads = 1;options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            report.update(ort_version=ort.__version__,providers=['CPUExecutionProvider'],maximum_simultaneous_ORT_sessions=1)
            def load(row):
                value = np.load(artifact(row),allow_pickle=False)
                require(str(value.dtype) == row['dtype'] and list(value.shape) == row['shape'] and
                    raw(value) == row['raw_sha256'] and value.flags.c_contiguous and np.isfinite(value).all(),'Pinned original tensor differs')
                return value
            all_pass = True
            for case in conf['inputs']:
                x = load(case['input']);x_sha = raw(x);baseline = None
                original = None
                if case['original_Torch_outputs_available']:
                    require(case['input']['raw_sha256'] == capture['artifacts']['frame0_input']['raw_sha256'],'Torch reference must use exact captured input')
                    original = []
                    for role in ('frame','global'):
                        row = dict(capture['artifacts'][role+'0_output'])
                        row['file'] = str(Path(conf['native_capture']['file']).parent/row['file'])
                        original.append(load(row))
                for variant in ('A','F','B'):
                    progress('CPU_gate_'+case['id']+'_'+variant)
                    started = time.perf_counter()
                    try:
                        session = ort.InferenceSession(str(graphs[variant]),options,providers=['CPUExecutionProvider'])
                        require(session.get_providers() == ['CPUExecutionProvider'] and
                            [(v.name,v.shape) for v in session.get_inputs()] == [('input',FRAME)] and
                            [(v.name,v.shape) for v in session.get_outputs()] == [('frame_output',FRAME),('global_output',GLOBAL)],
                            'Exact original pair ORT boundaries required')
                        actual = session.run(None,{'input':x})
                    finally:
                        session = None;gc.collect()
                    elapsed = time.perf_counter()-started
                    require(raw(x) == x_sha,'Original real input changed')
                    if variant == 'A':baseline = actual
                    checks,outputs = {},[]
                    for i,name in enumerate(('frame_output','global_output')):
                        require(list(actual[i].shape) == (FRAME if i == 0 else GLOBAL) and actual[i].dtype == np.float32
                            and np.isfinite(actual[i]).all(),'Finite original complete endpoint required')
                        outputs.append(save(case['id']+'-'+variant+'-'+name+'.npy',actual[i],np))
                        checks[name+'_vs_original_A_same_input'] = comparison(actual[i],baseline[i],np)
                        if original is not None:checks[name+'_vs_original_Torch_same_input'] = comparison(actual[i],original[i],np)
                    passed = all(c['fixed_gate_passed'] for c in checks.values());all_pass &= passed
                    report['checks'].append(dict(input_id=case['id'],variant=variant,outputs=outputs,comparisons=checks,
                        fixed_gate_passed=passed,session_dropped=True,CPU_session_and_run_wall_seconds=elapsed,
                        CPU_diagnostic_timing_not_NPU_or_performance_measurement=True))
                    progress('saved_'+case['id']+'_'+variant)
                    if variant != 'A':del actual
                del x,baseline,original;gc.collect()
            report.update(status='checked_original_basis_pair_fixed_FP32_gate_passed' if all_pass else 'checked_original_basis_pair_fixed_FP32_gate_failed',
                FP32_gate_passed=bool(all_pass),all6_calls_completed=True,ORT_sessions_dropped=True,
                original_numeric_gate_unchanged=True,strict_allclose_is_diagnostic=True,RKNN_GPU_or_quantization_executed=False)
    except BaseException as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        session = None;gc.collect()
        report['ORT_session_dropped'] = True
        report['source_and_inputs_unchanged'] = all(path.is_file() and sha(path) == expected for path,expected in watched.items())
        report['input_and_source_SHA_pins'] = {str(path):expected for path,expected in watched.items()}
        try:
            os.sched_setaffinity(0,affinity);os.environ.clear();os.environ.update(env)
            report.update(affinity_restored=set(os.sched_getaffinity(0)) == affinity,environment_restored=dict(os.environ) == env)
            require(report['affinity_restored'] and report['environment_restored'] and report['source_and_inputs_unchanged'],'Cleanup or source identity differs')
        except BaseException as error:report['cleanup_errors'].append(str(error));report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    if a.mode == 'export':return 0 if report['status'].startswith('exported_') else 1
    return 0 if report.get('FP32_gate_passed') and not report['cleanup_errors'] else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
