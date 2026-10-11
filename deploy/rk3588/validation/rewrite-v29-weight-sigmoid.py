#!/usr/bin/env python3
"""Replace only final confidence sigmoid with a bounded exponential identity.

Static ONNX edit only. No parameters are changed or inference performed.
For finite x: exp(min(x,0))/(1+exp(-abs(x))) is mathematically sigmoid(x).
Both exponential arguments are nonpositive, numerator is nonnegative and the
denominator is at least one. This is not output clipping. Actual board FP32
equivalence, domain checks and recurrent feedback still require validation.
"""
import argparse,copy,hashlib,json
from pathlib import Path
CAPTURE='87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
EXPORT='56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rewrite(model):
    import onnx
    import numpy as np
    m=copy.deepcopy(model);candidates=[n for n in m.graph.node if n.op_type=='Sigmoid' and list(n.output)==['weight']]
    if len(candidates)!=1 or len(candidates[0].input)!=1:raise ValueError('Exactly one terminal weight Sigmoid required')
    old=candidates[0];prefix='v29_weight_sigmoid/'
    names={v for n in m.graph.node for v in list(n.input)+list(n.output)}
    if any(x.startswith(prefix) for x in names):raise ValueError('Rewrite namespace already in use')
    h=onnx.helper;nt=onnx.numpy_helper
    def node(op,ins,name):return h.make_node(op,ins,[prefix+name],name=prefix+name)
    added=[h.make_node('Constant',[],[prefix+'zero'],name=prefix+'zero',value=nt.from_array(np.array(0,np.float32))),
           h.make_node('Constant',[],[prefix+'one'],name=prefix+'one',value=nt.from_array(np.array(1,np.float32))),
           node('Min',[old.input[0],prefix+'zero'],'nonpositive'),
           node('Abs',[old.input[0]],'abs'),node('Neg',[prefix+'abs'],'negative_abs'),
           node('Exp',[prefix+'nonpositive'],'numerator'),node('Exp',[prefix+'negative_abs'],'tail'),
           node('Add',[prefix+'one',prefix+'tail'],'denominator'),
           h.make_node('Div',[prefix+'numerator',prefix+'denominator'],['weight'],name=prefix+'weight')]
    old_bytes=old.SerializeToString();nodes=[]
    for n in m.graph.node:nodes.extend(added if n is old else [copy.deepcopy(n)])
    # Protobuf repeated containers need not preserve Python object identity.
    if len(nodes)!=len(model.graph.node)+8:
        nodes=[]
        for n in m.graph.node:nodes.extend(added if n.SerializeToString()==old_bytes else [copy.deepcopy(n)])
    del m.graph.node[:];m.graph.node.extend(nodes)
    assert [n.SerializeToString() for n in model.graph.initializer]==[n.SerializeToString() for n in m.graph.initializer]
    assert [n.SerializeToString() for n in model.graph.input]==[n.SerializeToString() for n in m.graph.input]
    assert [n.SerializeToString() for n in model.graph.output]==[n.SerializeToString() for n in m.graph.output]
    remaining=[n.SerializeToString() for n in m.graph.node if not n.name.startswith(prefix)]
    assert remaining==[n.SerializeToString() for n in model.graph.node if n.SerializeToString()!=old_bytes]
    onnx.checker.check_model(m,full_check=True)
    return m
def main():
    import onnx
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    assert sha(a.export_dir/'export.json')==EXPORT
    raw=json.loads((a.export_dir/'export.json').read_text());a.output_dir.mkdir(parents=True,exist_ok=False)
    result=dict(schema='v29_DPVO_Update_candidate_manifest',source_export_sha256=EXPORT,source_capture_sha256=CAPTURE,
        script_sha256=sha(__file__),host_model_inference=False,production_changed=False,models={},
        transformation='Only terminal weight Sigmoid replaced by Exp(Min(x,0))/(1+Exp(-Abs(x))). Original parameters, other nodes and I/O unchanged; no clipping or changed tolerance. No actual numeric acceptance inferred.')
    for key,meta in raw['models'].items():
        src=a.export_dir/meta['file'];assert sha(src)==meta['sha256'];model=onnx.load(src);new=rewrite(model)
        dest=a.output_dir/meta['file'];onnx.save(new,dest)
        result['models'][key]=dict(file=dest.name,sha256=sha(dest),bytes=dest.stat().st_size,source_onnx_sha256=meta['sha256'],inputs=meta['inputs'],outputs=meta['outputs'],role='stable_weight_sigmoid',original_initializers_byte_unchanged=True,other_nodes_byte_unchanged=True)
        assert sha(src)==meta['sha256']
    (a.output_dir/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'manifest_sha256':sha(a.output_dir/'manifest.json'),'models':{k:v['sha256'] for k,v in result['models'].items()}}))
if __name__=='__main__':main()
