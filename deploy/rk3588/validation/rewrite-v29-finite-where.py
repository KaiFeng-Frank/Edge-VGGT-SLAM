#!/usr/bin/env python3
"""Select finite Where branches using binary multiplication, static edit only.

This candidate requires the prior finite-padding rewrite. Replacing Where with
c*x+(1-c)*y is NOT safe if either branch may be nonfinite; the supplied recorded
inputs must be tested on board. No inference or numerical acceptance here.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nt
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rewrite(model):
    m=copy.deepcopy(model);before=[n.SerializeToString() for n in m.graph.initializer]
    constants=list(m.graph.initializer)+[a.t for n in m.graph.node if n.op_type=='Constant' for a in n.attribute if a.name=='value']
    for value in constants:
        a=nt.to_array(value)
        if a.dtype.kind=='f' and not np.isfinite(a).all():raise ValueError('Finite branch preparation required; nonfinite constant present')
    names={x for n in m.graph.node for x in list(n.input)+list(n.output)}
    if any(x.startswith('v29_finite_select/') for x in names):raise ValueError('Rewrite namespace reused')
    nodes=[];removed=[]
    for n in m.graph.node:
        if n.op_type!='Where':nodes.append(copy.deepcopy(n));continue
        if len(n.input)!=3 or len(n.output)!=1:raise ValueError('Three-input Where required')
        prefix='v29_finite_select/'+str(len(removed))+'/'
        nodes.extend([h.make_node('Cast',[n.input[0]],[prefix+'mask'],name=prefix+'mask',to=1),
            h.make_node('Constant',[],[prefix+'one'],name=prefix+'one',value=nt.from_array(np.array(1,np.float32))),
            h.make_node('Sub',[prefix+'one',prefix+'mask'],[prefix+'other'],name=prefix+'other'),
            h.make_node('Mul',[n.input[1],prefix+'mask'],[prefix+'yes'],name=prefix+'yes'),
            h.make_node('Mul',[n.input[2],prefix+'other'],[prefix+'no'],name=prefix+'no'),
            h.make_node('Add',[prefix+'yes',prefix+'no'],list(n.output),name=prefix+'select')])
        removed.append(n.name)
    if not removed:raise ValueError('No Where nodes found')
    del m.graph.node[:];m.graph.node.extend(nodes)
    assert [n.SerializeToString() for n in m.graph.initializer]==before
    assert [n.SerializeToString() for n in m.graph.input]==[n.SerializeToString() for n in model.graph.input]
    assert [n.SerializeToString() for n in m.graph.output]==[n.SerializeToString() for n in model.graph.output]
    assert [n.SerializeToString() for n in m.graph.node if not n.name.startswith('v29_finite_select/')]==[n.SerializeToString() for n in model.graph.node if n.op_type!='Where']
    onnx.checker.check_model(m,full_check=True)
    return m,removed
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--finite-dir',type=Path,required=True);p.add_argument('--finite-manifest-sha256',required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    assert sha(a.finite_dir/'manifest.json')==a.finite_manifest_sha256
    parent=json.loads((a.finite_dir/'manifest.json').read_text());result=copy.deepcopy(parent);result['models']={}
    result.update(script_sha256=sha(__file__),parent_manifest_sha256=a.finite_manifest_sha256,host_model_inference=False,
        transformation='Prior finite group-max padding plus all six Where nodes replaced with finite binary masked multiply/add. No weight/other-node/input/output change. Actual finite intermediate and FP32 gates must be checked on board.')
    a.output_dir.mkdir(exist_ok=False,parents=True)
    for key,meta in parent['models'].items():
        assert meta['role']=='finite_group_max_padding';src=a.finite_dir/meta['file'];assert sha(src)==meta['sha256']
        m,removed=rewrite(onnx.load(src));assert len(removed)==6
        out=a.output_dir/meta['file'];onnx.save(m,out)
        row=copy.deepcopy(meta);row.update(sha256=sha(out),bytes=out.stat().st_size,role='finite_arithmetic_selection',parent_onnx_sha256=meta['sha256'],removed_where_nodes=removed)
        result['models'][key]=row;assert sha(src)==meta['sha256']
    (a.output_dir/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'manifest_sha256':sha(a.output_dir/'manifest.json'),'models':{k:v['sha256'] for k,v in result['models'].items()}}))
if __name__=='__main__':main()
