#!/usr/bin/env python3
"""Expose selected existing tensors; static graph work, never inference."""
import argparse,copy,hashlib,json
from pathlib import Path
CAPTURE='87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
EXPORT='56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac'
TAPS=['/corr/corr.2/Add_output_0','/corr/corr.3/LayerNormalization_output_0',
      '/norm/LayerNormalization_output_0','/Add_3_output_0','/Gather_2_output_0',
      '/Where_output_0','/ReduceMax_output_0','/Exp_output_0','/Div_output_0',
      '/Add_5_output_0','/Gather_5_output_0','/Where_3_output_0','/Div_1_output_0',
      '/Add_7_output_0','/gru/gru.0/LayerNormalization_output_0','/gru/gru.2/LayerNormalization_output_0']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contract(x):
    dims=[d.dim_value for d in x.type.tensor_type.shape.dim]
    if not dims or not all(v>0 for v in dims):raise ValueError('Static positive output dimensions required')
    return dict(name=x.name,dtype=x.type.tensor_type.elem_type,shape=dims)
def main():
    import onnx
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    assert sha(a.export_dir/'export.json')==EXPORT
    source=json.loads((a.export_dir/'export.json').read_text());meta=source['models']['E864'];path=a.export_dir/meta['file'];assert sha(path)==meta['sha256']
    m=onnx.load(path);original=m.SerializeToString();original_nodes=[n.SerializeToString() for n in m.graph.node];original_params=[n.SerializeToString() for n in m.graph.initializer]
    inferred=onnx.shape_inference.infer_shapes(m);values={v.name:v for v in inferred.graph.value_info}
    for name in TAPS:
        assert name in values and values[name].type.tensor_type.elem_type==1
        contract(values[name]);m.graph.output.append(copy.deepcopy(values[name]))
    assert [n.SerializeToString() for n in m.graph.node]==original_nodes
    assert [n.SerializeToString() for n in m.graph.initializer]==original_params
    assert [contract(x) for x in m.graph.input]==meta['inputs']
    assert [contract(x) for x in m.graph.output[:3]]==meta['outputs']
    onnx.checker.check_model(m,full_check=True)
    a.output_dir.mkdir(parents=True,exist_ok=False);out=a.output_dir/'E864.onnx';onnx.save(m,out)
    result=dict(schema='v29_DPVO_Update_candidate_manifest',source_export_sha256=EXPORT,source_capture_sha256=CAPTURE,
        script_sha256=sha(__file__),host_model_inference=False,production_changed=False,
        transformation='Only graph.output entries added; original nodes, parameters, inputs and three outputs byte-preserved. Debug outputs can change compiler fusion and timing; results apply to this instrumented graph.',
        models=dict(E864=dict(file=out.name,sha256=sha(out),bytes=out.stat().st_size,source_onnx_sha256=meta['sha256'],
            inputs=[contract(x) for x in m.graph.input],outputs=[contract(x) for x in m.graph.output],role='baseline_debug_taps',
            added_outputs=TAPS,original_nodes_byte_unchanged=True,original_initializers_byte_unchanged=True)))
    (a.output_dir/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    assert sha(path)==meta['sha256'] and sha(a.export_dir/'export.json')==EXPORT
    print(json.dumps({'manifest_sha256':sha(a.output_dir/'manifest.json'),'onnx_sha256':sha(out),'outputs':len(m.graph.output)}))
if __name__=='__main__':main()
