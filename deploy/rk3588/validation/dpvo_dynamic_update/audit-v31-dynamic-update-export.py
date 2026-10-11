#!/usr/bin/env python3
"""Read-only ONNX metadata/initializer and saved-NPZ audit, never inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

STATIC_SHA='56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac'
CAPTURE_SHA='87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
ENTRY_SHA='cece09f65635c733d25469d7986d482dd83db22180edc6dd0fa6669f170704f1'
REWRITE_SHA='34bf40d845acecbadd43c95e01363370d133aab122451a105729802d5c378f37'
INPUTS=('net','inp','corr','previous_indices','previous_valid','following_indices','following_valid',
    'kk_indices','kk_valid','kk_inverse','ij_indices','ij_valid','ij_inverse')
OUTPUTS=('net_out','delta','weight')
SHAPES={name:[1,'E',384] for name in ('net','inp','net_out')}
SHAPES.update(corr=[1,'E',882],delta=[1,'E',2],weight=[1,'E',2])
for name in ('previous','following'):
    SHAPES[name+'_indices']=['E'];SHAPES[name+'_valid']=[1,'E',1]
for name,suffix in (('kk','kk'),('ij','ij')):
    SHAPES[name+'_indices']=SHAPES[name+'_valid']=['G'+suffix,'L'+suffix]
    SHAPES[name+'_inverse']=['E']


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda:file.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def meta(array):
    return dict(shape=list(array.shape),dtype=str(array.dtype),bytes=array.nbytes,
        tensor_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def artifact(folder,item):
    require(Path(item['file']).name==item['file'],'Artifact basename required')
    path=folder/item['file'];require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Artifact SHA/bytes differ')
    with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
    require({k:meta(v) for k,v in arrays.items()}==item['arrays'],'Saved raw array metadata differs')
    return arrays


def contracts(model):
    out=[]
    for value in list(model.graph.input)+list(model.graph.output):
        tensor=value.type.tensor_type
        shape=[d.dim_param if d.HasField('dim_param') else d.dim_value for d in tensor.shape.dim]
        dtype=7 if value.name.endswith('_indices') or value.name.endswith('_inverse') else 1
        require(tensor.elem_type==dtype and shape==SHAPES[value.name],'Dynamic dimension/fixed geometry or dtype differs')
        out.append(dict(name=value.name,dtype=tensor.elem_type,shape=shape))
    require([v.name for v in model.graph.input]==list(INPUTS) and [v.name for v in model.graph.output]==list(OUTPUTS),
        'All thirteen inputs/three outputs must remain in order')
    require(all(any(v.name in n.input for n in model.graph.node) for v in model.graph.input),'Unused topology input')
    return out[:13],out[13:]


def initializer_uses(model):
    from onnx import numpy_helper
    result=[]
    for tensor in model.graph.initializer:
        require(not tensor.external_data and tensor.data_location!=1,'External initializers prohibited')
        value=numpy_helper.to_array(tensor)
        require(value.dtype==np.float32 and np.isfinite(value).all(),'Original initializer dtype/finite differs')
        uses=sorted((n.name,n.op_type,index) for n in model.graph.node for index,name in enumerate(n.input) if name==tensor.name)
        require(bool(uses),'Unused original initializer')
        result.append((str(value.dtype),tuple(value.shape),hashlib.sha256(value.tobytes(order='C')).hexdigest(),tuple(uses)))
    return Counter(result)


def shape_routes(model):
    """Specific observed export pattern: two same-data reshapes and six live fills."""
    from onnx import numpy_helper
    producers={name:node for node in model.graph.node for name in node.output}
    reshapes=[];fills=[];integer_literals=[]
    for node in model.graph.node:
        if node.op_type=='Reshape':
            parent=producers[node.input[1]]
            require(parent.op_type=='Shape' and list(parent.input)==[node.input[0]],'Reshape uses frozen or indirect shape')
            reshapes.append(dict(node=node.name,shape_from_same_data=True))
        elif node.op_type=='ConstantOfShape':
            parent=producers[node.input[0]]
            require(parent.op_type=='Shape','Shape fill is not dynamically derived')
            value=numpy_helper.to_array(next(a.t for a in node.attribute if a.name=='value'))
            fills.append(dict(node=node.name,shape_source=parent.input[0],fill='negative_infinity' if np.isneginf(value).all() else value.tolist()))
        elif node.op_type=='Constant':
            attributes=[a.t for a in node.attribute if a.name=='value']
            require(len(attributes)==1,'Unreviewed Constant representation')
            value=numpy_helper.to_array(attributes[0])
            if np.issubdtype(value.dtype,np.integer):
                require(set(value.ravel().tolist())<={-1,0,2},'Unexpected fixed edge/group/shape literal')
                integer_literals.append(dict(node=node.name,value=value.tolist()))
    require(len(reshapes)==2 and len(fills)==6,'Observed dynamic shape-route structure changed')
    return dict(reshapes=reshapes,dynamic_shape_fills=fills,integer_literals=integer_literals,
        frozen_E96_or_group_dimension_literal_found=False,
        limit='Checks this exported shape-route pattern; does not prove runtime acceptance for every dimension.')


def terminal_only(raw,final):
    old=[n for n in raw.graph.node if n.op_type=='Sigmoid' and list(n.output)==['weight']]
    require(len(old)==1,'Terminal original weight Sigmoid missing')
    for attr in ('initializer','input','output'):
        require([x.SerializeToString() for x in getattr(raw.graph,attr)]==
            [x.SerializeToString() for x in getattr(final.graph,attr)],'Terminal rewrite changed '+attr)
    prefix='v29_weight_sigmoid/'
    require([n.SerializeToString() for n in raw.graph.node if n.SerializeToString()!=old[0].SerializeToString()]==
        [n.SerializeToString() for n in final.graph.node if not n.name.startswith(prefix)],'Nonterminal node changed')
    added=[n for n in final.graph.node if n.name.startswith(prefix)]
    require([n.op_type for n in added]==['Constant','Constant','Min','Abs','Neg','Exp','Exp','Add','Div'],
        'Stable sigmoid recipe differs')
    require(len(final.graph.node)==len(raw.graph.node)+8,'Unexpected added node count')


def audit(args):
    import onnx
    root=Path(args.export_dir);manifest_path=root/'manifest.json'
    require(sha(manifest_path)==args.export_sha256,'Dynamic manifest pin differs')
    manifest=json.loads(manifest_path.read_text())
    require(manifest['schema']=='v31_DPVO_dynamic_Update_export' and
        manifest['status']=='exported_dynamic_ONNX_numerical_validation_pending' and
        manifest['script_sha256']==ENTRY_SHA,'Expected frozen dynamic export required')
    require(manifest['source_pins']==manifest['source_hashes_after'] and manifest['sources_and_inputs_unchanged'] is True and
        manifest['cleanup_errors']==[] and manifest['affinity_restored'] is True,'Board immutability/cleanup failed')
    capture_path=Path(args.capture_dir)/'capture.json';static_path=Path(args.static_export_dir)/'export.json'
    require(sha(capture_path)==CAPTURE_SHA and sha(static_path)==STATIC_SHA,'Original capture/static export pins differ')
    capture=json.loads(capture_path.read_text());static=json.loads(static_path.read_text())
    require(len(manifest['original_weights_before'])==94 and len(manifest['original_Update_weights_before'])==50 and
        manifest['original_weights_before']==manifest['original_weights_after']==capture['original_weights_before']==static['original_weights_before'] and
        manifest['original_Update_weights_before']==manifest['original_Update_weights_after'] and
        manifest['original_weights_unchanged'] is True and manifest['original_Update_weights_unchanged'] is True,'Original weights metadata differs')
    graphs={}
    for key in ('raw','final'):
        item=manifest['models'][key];path=root/item['file']
        require(Path(item['file']).name==item['file'] and sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],'Graph file pin differs')
        graph=onnx.load(path,load_external_data=False);onnx.checker.check_model(graph,full_check=True)
        require([(o.domain,o.version) for o in graph.opset_import]==[('',17)],'ONNX opset differs')
        inputs,outputs=contracts(graph)
        require(inputs==item['inputs'] and outputs==item['outputs'] and len(graph.graph.node)==item['nodes'] and
            len(graph.graph.initializer)==item['initializers']==50,'Graph metadata differs')
        graphs[key]=graph
    parent=static['models']['E96'];path=Path(args.static_export_dir)/parent['file']
    require(sha(path)==parent['sha256'] and path.stat().st_size==parent['bytes'],'Original static graph pin differs')
    original=onnx.load(path,load_external_data=False);onnx.checker.check_model(original,full_check=True)
    require(initializer_uses(original)==initializer_uses(graphs['raw'])==initializer_uses(graphs['final']),
        'Parameter bits/shapes/dtypes or semantic consumers differ from original static export')
    routes=shape_routes(graphs['raw']);terminal_only(graphs['raw'],graphs['final'])
    require(manifest['rewrite']['source_sha256']==REWRITE_SHA and manifest['rewrite']['terminal_weight_only'] is True,
        'Frozen terminal rewrite source differs')
    example=manifest['trace_example'];require(example['case']=='first_motion_0_to_1_identity' and example['step']==0 and example['edges']==96,'Trace example identity differs')
    inputs=artifact(root,example['inputs_artifact']);reference=artifact(root,example['reference_artifact']);ids=artifact(root,example['graph_artifact'])
    state=next(c for c in capture['scenarios'] if c['name']==example['case'])['states'][0]
    old_inputs=artifact(Path(args.capture_dir),state['inputs']);old_outputs=artifact(Path(args.capture_dir),state['outputs'])
    for new,old in (('net','net'),('inp','ctx'),('corr','corr')):
        require(meta(inputs[new])==meta(old_inputs[old]),'Original trace input raw bytes differ')
    require({k:meta(v) for k,v in ids.items()}=={k:meta(old_inputs[k]) for k in ('ii','jj','kk')},'Original graph example differs')
    for new,old in (('net_out','net'),('delta','delta'),('weight','weight')):
        require(meta(reference[new])==meta(old_outputs[old]),'Original trace reference raw bytes differ')
    require(manifest['trace_example_wrapper_gate_passed'] is True and all(c['rtol']==1e-4 and c['atol']==1e-5 and
        c['failed_elements']==0 and c['gate_passed'] is True for c in manifest['trace_example_wrapper_checks'].values()),'Reported trace wrapper gate failed/changed')
    require(manifest['ORT_numeric_gate_exercised'] is False and manifest['dynamic_shape_numeric_gate_exercised'] is False and
        manifest['RKNN_dynamic_shapes_support']=='not_tested','Export overstates dynamic numerical/runtime validation')
    return dict(schema='v31_dynamic_Update_independent_static_review',status='passed',failed=0,
        audit_script_sha256=sha(__file__),manifest_sha256=args.export_sha256,static_export_sha256=STATIC_SHA,
        original_capture_sha256=CAPTURE_SHA,model_sha256={k:manifest['models'][k]['sha256'] for k in graphs},
        all_13_dynamic_inputs_used=True,all_3_dynamic_outputs_checked=True,fixed_batch=1,fixed_net_channels=384,
        fixed_corr_channels=882,fixed_delta_weight_channels=2,dynamic_symbols=['E','Gkk','Lkk','Gij','Lij'],
        original_neural_parameter_tensors=94,original_Update_parameter_tensors=50,
        initializer_values_and_semantic_consumers_equal_original_static=True,
        anonymous_initializer_names='Torch export renumbered anonymous MatMul tensor names; dtype/shape/raw bytes and consumer node/op/slot unchanged.',
        raw_nodes=len(graphs['raw'].graph.node),final_nodes=len(graphs['final'].graph.node),
        terminal_stable_sigmoid_only=True,raw_final_initializers_and_IO_byte_identical=True,
        shape_routes=routes,source_before_after_and_cleanup='Board reported equality verified; not all board-only source/checkpoint files rehashed on host.',
        example_input_and_golden_raw_bytes_equal_original=True,reported_trace_example_wrapper_gate_checked=True,
        standard_ONNX_checker_passed=True,host_execution='ONNX checker/metadata and initializer bytes plus saved NumPy arrays only. No model, ORT, ReferenceEvaluator or BA execution.',
        dynamic_numeric_runtime_exercised=False,RKNN_dynamic_shapes_support='not_tested',
        accuracy_or_realtime_validated=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('export-dir','static-export-dir','capture-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--export-sha256',required=True);args=p.parse_args(argv)
    require(not args.output.exists(),'New output required; preserve previous audit receipts')
    try:result=audit(args)
    except Exception as error:result=dict(schema='v31_dynamic_Update_independent_static_review',status='failed',error=repr(error),audit_script_sha256=sha(__file__))
    with args.output.open('x') as file:json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps({k:result.get(k) for k in ('status','failed','error')}))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
