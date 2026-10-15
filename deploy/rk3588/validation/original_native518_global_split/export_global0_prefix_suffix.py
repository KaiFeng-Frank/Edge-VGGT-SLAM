#!/usr/bin/env python3
"""Static exact cuts of original native global0: prefix QKV and residual suffix.

No model, ORT, SDK or GPU execution. The existing original SDPA head-part
graphs are reused; each cut retains exact original bytes of its initializers.
The suffix consumes the original input residual and full attention_values.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import traceback
import types

HEAD_EXPORTER_SHA='8ebf9183d8b04cf3769f5de0e873ef912551b88ce26dac56ab66db883329573c'
HEAD_EXPORT_SHA='e09acb577228c66d3bbae9a14ce5f22115f8c1f4ea7b128cafbbc61709dcaa6e'
PARENT_SHA='c44094793c3b590c6066cbdd634cf0c4d8bd5b75987c8b491db7b17b2201f55d'
PARENT_BYTES=51469885
SCHEMA='v114_original_native518_global0_prefix_suffix_ONNX_export'
SUCCESS='exported_original_global0_exact_cuts_not_FP32_or_NPU_accepted'
FRAME=(1,2082,1024)
BHSD=(1,16,2082,64)
PREFIX_OUTPUTS=('query_rope','key_rope','value')
SUFFIX_INPUTS=('input','attention_values')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def pin(path,digest):
    path=Path(path).resolve(strict=True)
    require(sha(path)==digest,'Pinned original source/artifact differs: '+str(path));return path


def atomic(path,row):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');temp.replace(path)


def helper(path):
    path=pin(path,HEAD_EXPORTER_SHA);m=types.ModuleType('v114_frozen_head_extractor');m.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def split_graphs(parent,capture,E):
    import onnx
    from onnx import helper as H,TensorProto as T
    tails,tail_proof=E.extract_models(parent,capture['parameters'])
    original=parent.SerializeToString();g=parent.graph
    require([(v.name,[d.dim_value for d in v.type.tensor_type.shape.dim],v.type.tensor_type.elem_type)
        for v in (*g.input,*g.output)]==[('input',list(FRAME),T.FLOAT),('output',list(FRAME),T.FLOAT)],
        'Exact complete original global0 IO required')
    nodes={n.name:n for n in g.node}
    producers={name:n.name for n in g.node for name in n.output}
    require(len(nodes)==len(g.node) and len(producers)==sum(len(n.output) for n in g.node),'Unique original dataflow required')
    initializers={v.name for v in g.initializer}
    def closure(outputs,boundaries):
        selected=set();visiting=set()
        def visit(value):
            if not value or value in boundaries or value in initializers:return
            require(value in producers,'Unbound original tensor: '+value)
            name=producers[value]
            if name in selected:return
            require(name not in visiting,'Original graph dataflow cycle')
            visiting.add(name)
            for item in nodes[name].input:visit(item)
            visiting.remove(name);selected.add(name)
        for output in outputs:visit(output)
        return selected
    prefix=closure(PREFIX_OUTPUTS,{'input'})
    suffix=closure(('output',),set(SUFFIX_INPUTS))
    attention=set(E.NODE_NAMES)
    require(not prefix&suffix and not prefix&attention and not suffix&attention and
        prefix|suffix|attention==set(nodes),'The two cuts plus original five attention nodes must cover all53 nodes exactly once')
    inferred=onnx.shape_inference.infer_shapes(parent,strict_mode=True,check_type=True,data_prop=True)
    info={v.name:v for v in (*inferred.graph.input,*inferred.graph.output,*inferred.graph.value_info)}
    for name in (*PREFIX_OUTPUTS,'attention_values'):
        require([d.dim_value for d in info[name].type.tensor_type.shape.dim]==list(BHSD) and
            info[name].type.tensor_type.elem_type==T.FLOAT,'Original complete BHSD boundary differs: '+name)
    specs=[('prefix',prefix,('input',),PREFIX_OUTPUTS),('suffix',suffix,SUFFIX_INPUTS,('output',))]
    used={key:{value for n in g.node if n.name in selected for value in n.input if value in initializers}
          for key,selected,_,_ in specs}
    tail_initializers={v.name for v in tails['heads16'].graph.initializer}
    # Preserve any already-unused parent metadata constants once in the suffix,
    # rather than silently dropping original provenance or duplicating weights.
    unused=initializers-used['prefix']-used['suffix']-tail_initializers
    used['suffix'].update(unused)
    models={};rows=[]
    for key,selected,inputs,outputs in specs:
        model=copy.deepcopy(parent)
        del model.graph.node[:];model.graph.node.extend([nodes[n.name] for n in g.node if n.name in selected])
        del model.graph.initializer[:]
        model.graph.initializer.extend([v for v in g.initializer if v.name in used[key]])
        del model.graph.input[:];model.graph.input.extend([copy.deepcopy(info[name]) for name in inputs])
        del model.graph.output[:];model.graph.output.extend([copy.deepcopy(info[name]) for name in outputs])
        del model.graph.value_info[:]
        model=onnx.shape_inference.infer_shapes(model,strict_mode=True,check_type=True,data_prop=True)
        onnx.checker.check_model(model,full_check=True)
        require([n.SerializeToString() for n in model.graph.node]==[n.SerializeToString() for n in g.node if n.name in selected] and
            [v.SerializeToString() for v in model.graph.initializer]==[v.SerializeToString() for v in g.initializer if v.name in used[key]],
            'Original node or selected initializer protobufs changed in '+key)
        models[key]=model
        def bindings(values):
            return [dict(index=i,name=v.name,shape=[d.dim_value for d in v.type.tensor_type.shape.dim],dtype='float32')
                for i,v in enumerate(values)]
        rows.append(dict(key=key,node_count=len(model.graph.node),initializer_count=len(model.graph.initializer),
            original_node_names=[n.name for n in model.graph.node],input_bindings=bindings(model.graph.input),
            output_bindings=bindings(model.graph.output)))
    recovered=copy.deepcopy(parent);del recovered.graph.node[:]
    pieces={n.name:n for model in models.values() for n in model.graph.node}
    pieces.update({n.name:n for n in tails['heads16'].graph.node})
    require(set(pieces)==set(nodes),'Original union node identities differ')
    recovered.graph.node.extend([pieces[n.name] for n in g.node])
    initializer_pieces={}
    for model in (*models.values(),tails['heads16']):
        for value in model.graph.initializer:
            if value.name in initializer_pieces:
                require(initializer_pieces[value.name].SerializeToString()==value.SerializeToString(),
                    'Shared original initializer bytes differ')
            initializer_pieces[value.name]=value
    require(set(initializer_pieces)==initializers,'The full original initializer union differs')
    del recovered.graph.initializer[:]
    recovered.graph.initializer.extend([initializer_pieces[v.name] for v in g.initializer])
    require(recovered.SerializeToString()==original,'Reassembled original whole parent protobuf differs')
    proof=dict(original_parent_nodes=53,original_parent_initializers=44,
        original_node_union_exactly_once=True,whole_parent_reassembled_protobuf_equal=True,
        original_initializer_values_and_protobufs_preserved_in_their_cuts=True,
        full_initializer_union_including_original_SDPA_protobuf_equal=True,
        shared_initializer_names=sorted(used['prefix']&used['suffix']),
        already_unused_parent_initializers_preserved_once_in_suffix=sorted(unused),
        all18_original_learned_parameters_verified=tail_proof['all18_parent_learned_parameters_match_original_capture'],
        original_attention_nodes=list(E.NODE_NAMES),original_residual_input_preserved_in_suffix=True,
        original_suffix_transpose_projection_norm_GELU_FC2K4_and_residuals_preserved=True,
        full2082_queries_keys_16_heads_64_dimensions=True,attention_scale=.125,
        prefix_output_order=list(PREFIX_OUTPUTS),suffix_input_order=list(SUFFIX_INPUTS),
        original_SDPA_head_partition_proof=tail_proof,extra_output_taps_in_full_parent=False,
        added_external_IO_may_change_compiler_fusion_and_FP16_rounding=True,
        FP32_or_compiled_NPU_quality_or_performance_accepted=False)
    return models,rows,proof


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-onnx',type=Path,required=True);p.add_argument('--parent-sha256',required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--capture-sha256',required=True)
    p.add_argument('--head-exporter-source',type=Path,required=True);p.add_argument('--head-exporter-sha256',required=True)
    p.add_argument('--head-export',type=Path,required=True);p.add_argument('--head-export-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    r=dict(schema=SCHEMA,status='failed',script_sha256=sha(__file__),original_model_only=True,
        model_ORT_SDK_GPU_execution=False,component_only=True,candidate_accepted=False,cleanup_errors=[])
    watched={str(Path(__file__).resolve()):r['script_sha256']}
    try:
        require(a.parent_sha256==PARENT_SHA and a.head_exporter_sha256==HEAD_EXPORTER_SHA and
            a.head_export_sha256==HEAD_EXPORT_SHA,'Exact parent/head graph source lineage required')
        E=helper(a.head_exporter_source);watched[str(Path(a.head_exporter_source).resolve())]=HEAD_EXPORTER_SHA
        require(a.capture_sha256==E.CAPTURE_SHA,'Genuine native capture required')
        parent=pin(a.parent_onnx,PARENT_SHA);require(parent.stat().st_size==PARENT_BYTES,'Parent graph byte size differs')
        cp,c=E.capture_receipt(a.capture);hp=pin(a.head_export,HEAD_EXPORT_SHA);h=json.loads(hp.read_text())
        require(h['schema']==E.SCHEMA and h['status']==E.SUCCESS and h['script_sha256']==HEAD_EXPORTER_SHA and
            h['parent']['sha256']==PARENT_SHA and h['capture']['sha256']==E.CAPTURE_SHA and h['parts']==E.PARTS and
            h['source_and_inputs_unchanged'] and not h['cleanup_errors'],'Exact already-gated original head extraction required')
        watched.update({str(parent):PARENT_SHA,str(cp):E.CAPTURE_SHA,str(hp):HEAD_EXPORT_SHA})
        import onnx
        models,rows,proof=split_graphs(onnx.load_model(parent),c,E)
        for row in rows:
            file=out/(row['key']+'.onnx');file.write_bytes(models[row['key']].SerializeToString())
            row.update(file=file.name,bytes=file.stat().st_size,sha256=sha(file))
        r.update(status=SUCCESS,checkpoint=E.CHECKPOINT,parent=dict(file=str(parent),bytes=PARENT_BYTES,sha256=PARENT_SHA),
            capture=dict(file=str(cp),sha256=E.CAPTURE_SHA),head_export=dict(file=str(hp),sha256=HEAD_EXPORT_SHA),
            head_exporter_source_sha256=HEAD_EXPORTER_SHA,models=rows,parts=copy.deepcopy(E.PARTS),proof=proof,
            original_parameter_metadata=c['parameters'],original_source_hashes=c['source_hashes'],
            original_SDPA_arguments=c['SDPA_arguments'],source_static_or_NPU_precision_not_inferred_from_partition=True)
    except BaseException as error:r.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        errors=[]
        for file,digest in watched.items():
            try:require(sha(file)==digest,'Pinned file changed: '+file)
            except BaseException as error:errors.append(str(error))
        r['source_and_inputs_unchanged']=not errors;r['cleanup_errors'].extend(errors)
        if errors:r['status']='failed'
        atomic(out/'export.json',r)
    print(json.dumps({k:r.get(k) for k in ('status','models','error')}),flush=True)
    return 0 if r['status']==SUCCESS else 1


if __name__=='__main__':raise SystemExit(main())
