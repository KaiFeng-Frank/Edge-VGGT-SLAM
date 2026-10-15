#!/usr/bin/env python3
"""Static extraction of original full2082 SDPA, with 16/6/5 independent heads.

This is an operator diagnostic. Original post-QK-normalization/post-RoPE Q/K
and original V are the three inputs; no learned parameter or layer is replaced.
The full original model remains outside this component's numerical acceptance.
No ORT, model, SDK or GPU execution is performed by this exporter.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import traceback

PARENT_SHA = 'c44094793c3b590c6066cbdd634cf0c4d8bd5b75987c8b491db7b17b2201f55d'
PARENT_BYTES = 51469885
CAPTURE_SHA = 'e4ffaea2b11df9ac558c119cb4854d490b0030c02b7706eee344b8894cefa686'
CAPTURE_SOURCE_SHA = '985d0c1daa7bcacd2d0ea6b184f396c35e6c9c3c500c8ee73642db001722a31c'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
SCHEMA = 'v112_original_native518_SDPA_head_parts_ONNX_export'
SUCCESS = 'exported_original_SDPA_parts_not_FP32_or_NPU_accepted'
N, DIM, HEADS = 2082, 64, 16
INPUT_NAMES = ('query_rope', 'key_rope', 'value')
NODE_NAMES = ('key_transpose', 'attention_scores', 'attention_scaled_scores',
              'attention_probabilities', 'attention_values')
PARTS = [dict(index=0,start=0,end=6,heads=6,model_key='heads6',core_mask=1),
         dict(index=1,start=6,end=11,heads=5,model_key='heads5',core_mask=2),
         dict(index=2,start=11,end=16,heads=5,model_key='heads5',core_mask=4)]


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20), b''): h.update(chunk)
    return h.hexdigest()


def pin(path, digest):
    path = Path(path).resolve(strict=True)
    require(len(digest)==64 and sha(path)==digest, 'Pinned file differs: '+str(path))
    return path


def atomic(path, result):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(path)


def capture_receipt(path):
    path=pin(path,CAPTURE_SHA);r=json.loads(path.read_text())
    require(r['schema']=='v103_original_native518_global0_attention_capture' and
        r['status']=='completed_original_native518_global0_attention_capture' and r['completed'] and
        r['script_sha256']==CAPTURE_SOURCE_SHA and r['checkpoint']==CHECKPOINT and
        r['logical_shape']==[1,N,1024] and r['image_shape']==[2,3,392,518] and
        r['original_block']=='global0' and r['original_SDPA_wrapper_forwards_unchanged'] and
        not r['observer_return_changes_model_output'] and r['original_whole_block_comparison']['byte_equal'] and
        r['original_parameters_verified']==18 and r['original_rope_tables_bitwise_equal_to_native_reference'] and
        r['model_state_unchanged'] and r['input_and_parameters_unchanged'] and r['SDPA_function_restored'] and
        r['hooks_removed'] and r['source_and_inputs_unchanged'] and r['affinity_restored'] and
        r['environment_restored'] and not r['cleanup_errors'], 'Clean genuine native original attention capture required')
    require(r['SDPA_arguments']==dict(scale_argument=None,effective_scale=.125,mask=None,dropout_p=0.,
        is_causal=False,heads=16,head_dim=64,keys=N,queries=N,dtype='float32'), 'Original complete SDPA equation differs')
    return path,r


def load_capture_arrays(path, receipt):
    import numpy as np
    arrays={};pins={}
    for name,shape in [('q_post_rope',[1,16,N,DIM]),('k_post_rope',[1,16,N,DIM]),
                       ('v',[1,16,N,DIM]),('attention_tail',[1,16,N,DIM]),('proj_input',[1,N,1024])]:
        row=receipt['artifacts'][name]
        require(Path(row['file']).name==row['file'], 'Direct captured NPY filename required')
        file=pin(path.parent/row['file'],row['sha256']);pins[str(file)]=row['sha256']
        value=np.load(file,allow_pickle=False)
        require(file.stat().st_size==row['bytes'] and list(value.shape)==row['shape']==shape and
            str(value.dtype)==row['dtype']=='float32' and value.nbytes==row['tensor_bytes'] and
            row['finite'] and np.isfinite(value).all() and value.flags.c_contiguous and
            hashlib.sha256(value.tobytes()).hexdigest()==row['raw_sha256'], 'Original capture array differs: '+name)
        arrays[name]=value
    expected=np.ascontiguousarray(arrays['attention_tail'].transpose(0,2,1,3).reshape(1,N,1024))
    require(np.array_equal(expected,arrays['proj_input']), 'Original consumer layout is not exact BHSD transpose/reshape')
    return arrays,pins


def extract_models(parent, parameters):
    import numpy as np
    import onnx
    from onnx import helper as H, numpy_helper as A, TensorProto as T
    onnx.checker.check_model(parent,full_check=True)
    require(len(parent.graph.node)==53 and len(parent.graph.initializer)==44 and
        len(parent.graph.input)==len(parent.graph.output)==1, 'Pinned complete global0 K4 parent shape differs')
    nodes={n.name:n for n in parent.graph.node};inits={v.name:v for v in parent.graph.initializer}
    require(len(nodes)==53 and len(inits)==44, 'Unique original nodes/initializers required')
    expected=[('Transpose',['key_rope']),('MatMul',['query_rope','key_transpose']),
              ('Mul',['attention_scores','attention_scale']),('Softmax',['attention_scaled_scores']),
              ('MatMul',['attention_probabilities','value'])]
    for name,(op,inputs) in zip(NODE_NAMES,expected):
        node=nodes[name]
        require(node.op_type==op and not node.domain and list(node.input)==inputs and list(node.output)==[name],
            'Exact original attention node differs: '+name)
    require({a.name:H.get_attribute_value(a) for a in nodes['key_transpose'].attribute}=={'perm':[0,1,3,2]} and
        {a.name:H.get_attribute_value(a) for a in nodes['attention_probabilities'].attribute}=={'axis':-1},
        'Original transpose/complete-key normalization differs')
    scale=inits['attention_scale'];value=A.to_array(scale)
    require(value.shape==() and value.dtype==np.float32 and value.item()==.125, 'Retain original scale')
    require(len(parameters)==18, 'All eighteen original global0 learned values required in parent lineage')
    linear={'attn.qkv.weight','attn.proj.weight','mlp.fc1.weight','mlp.fc2.weight'}
    for name,row in parameters.items():
        stored=name+'_transposed' if name in linear else name
        if name=='mlp.fc2.weight':value=np.concatenate([A.to_array(inits[stored+'.part'+str(i)]) for i in range(4)],axis=0).T
        else:
            value=A.to_array(inits[stored]);value=value.T if stored!=name else value
        value=np.ascontiguousarray(value)
        require(dict(shape=list(value.shape),dtype=str(value.dtype),tensor_bytes=value.nbytes,
            raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())==row and np.isfinite(value).all(),
            'Original learned value differs: '+name)
    templates=[copy.deepcopy(nodes[name]) for name in NODE_NAMES]
    models={}
    for key,heads in [('heads16',16),('heads6',6),('heads5',5)]:
        shape=[1,heads,N,DIM]
        graph=H.make_graph(copy.deepcopy(templates),'original_global0_SDPA_'+key,
            [H.make_tensor_value_info(name,T.FLOAT,shape) for name in INPUT_NAMES],
            [H.make_tensor_value_info('attention_values',T.FLOAT,shape)],[copy.deepcopy(scale)])
        model=H.make_model(graph,opset_imports=copy.deepcopy(parent.opset_import),producer_name='original_VGGT_SDPA_partition_diagnostic')
        model.ir_version=parent.ir_version
        model=onnx.shape_inference.infer_shapes(model,strict_mode=True,check_type=True,data_prop=True)
        onnx.checker.check_model(model,full_check=True)
        require([n.SerializeToString() for n in model.graph.node]==[n.SerializeToString() for n in templates] and
            model.graph.initializer[0].SerializeToString()==scale.SerializeToString(),
            'Extraction must preserve all original attention node/scale protobufs')
        shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in (*model.graph.input,*model.graph.output,*model.graph.value_info)}
        require(shapes['attention_scores']==shapes['attention_probabilities']==[1,heads,N,N] and
            shapes['attention_values']==shape,'Full query/key/head dimension contract differs')
        models[key]=model
    require(sum(p['heads'] for p in PARTS)==HEADS and PARTS[0]['start']==0 and PARTS[-1]['end']==HEADS and
        all(a['end']==b['start'] for a,b in zip(PARTS,PARTS[1:])), 'Ordered head partition must cover all16 exactly')
    proof=dict(original_attention_node_protobufs_unchanged=True,original_scale_initializer_protobuf_unchanged=True,
        all18_parent_learned_parameters_match_original_capture=True,learned_parameters_inside_SDPA=0,
        original_full_block_not_replaced_by_this_component=True,complete_key_rows=N,complete_query_rows=N,
        head_dim=DIM,heads=HEADS,scale=.125,softmax_axis=-1,head_ranges=copy.deepcopy(PARTS),
        quantization_or_FP16_rounding_not_validated=True,namespace_or_layer_omission_used=False)
    return models,proof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-onnx',type=Path,required=True);p.add_argument('--parent-sha256',required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--capture-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(schema=SCHEMA,status='failed',script_sha256=sha(__file__),original_model_only=True,
        component_only=True,model_ORT_SDK_or_GPU_execution=False,candidate_accepted=False,cleanup_errors=[])
    watched={str(Path(__file__).resolve()):r['script_sha256']}
    try:
        require(a.parent_sha256==PARENT_SHA and a.capture_sha256==CAPTURE_SHA,'Exact original lineage required')
        parent=pin(a.parent_onnx,PARENT_SHA);require(parent.stat().st_size==PARENT_BYTES,'Parent graph size differs')
        cp,capture=capture_receipt(a.capture);watched.update({str(parent):PARENT_SHA,str(cp):CAPTURE_SHA})
        import onnx
        models,proof=extract_models(onnx.load_model(parent),capture['parameters'])
        rows=[]
        for key,model in models.items():
            file=out/(key+'.onnx');file.write_bytes(model.SerializeToString());heads=int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
            rows.append(dict(key=key,file=file.name,bytes=file.stat().st_size,sha256=sha(file),heads=heads,
                input_bindings=[dict(index=i,name=n,shape=[1,heads,N,DIM],dtype='float32') for i,n in enumerate(INPUT_NAMES)],
                output_bindings=[dict(index=0,name='attention_values',shape=[1,heads,N,DIM],dtype='float32')]))
        r.update(status=SUCCESS,checkpoint=CHECKPOINT,parent=dict(file=str(parent),bytes=PARENT_BYTES,sha256=PARENT_SHA),
            capture=dict(file=str(cp),sha256=CAPTURE_SHA),models=rows,parts=copy.deepcopy(PARTS),proof=proof,
            original_source_hashes=capture['source_hashes'],original_SDPA_arguments=capture['SDPA_arguments'])
    except BaseException as error:r.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        r['source_and_inputs_unchanged']=all(sha(file)==digest for file,digest in watched.items())
        if not r['source_and_inputs_unchanged']:r['status']='failed';r['cleanup_errors'].append('Pinned file changed')
        atomic(out/'export.json',r)
    print(json.dumps({k:r.get(k) for k in ('status','models','error')}),flush=True)
    return 0 if r['status']==SUCCESS else 1


if __name__=='__main__':raise SystemExit(main())
