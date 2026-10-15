#!/usr/bin/env python3
"""Distinct-axis generated mixed-layout transport, no SDK/NN/GPU/board calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types
import numpy as np

ROOT=Path(__file__).resolve().parent
PIN='9b35689eed8f282b7f77022721f83a80e84b7abd046d6b589ebd2e54bbbad820'
GLOBAL=(1,2082,1024);BHSD=(1,16,2082,64)


def attr(name,index,shape,fmt):
    api=(1,2082,64,16) if len(shape)==4 and fmt==1 else shape;n=int(np.prod(shape))
    return dict(name=name,index=index,shape=list(api),n_elems=n,size=n*2,format=fmt,type=1,
        quantization_type=2,zero_point=0,scale=1.,pass_through=0)


class TransportFixture:
    def __init__(self,shapes,names,formats,output_shapes,output_names,output_formats,seeds,bad=None):
        self.input_shapes=list(shapes);self.output_shapes=list(output_shapes);self.seeds=seeds
        inputs=[attr(name,i,shape,fmt) for i,(name,shape,fmt) in enumerate(zip(names,shapes,formats))]
        outputs=[attr(name,i,shape,fmt) for i,(name,shape,fmt) in enumerate(zip(output_names,output_shapes,output_formats))]
        if bad=='rank3_format':inputs[0]['format']=1
        if bad=='rank4_shape':outputs[0]['shape']=[1,64,2082,16]
        if bad=='packed':outputs[0]['format']=2
        self.api={1:inputs,2:outputs};self.context=None
        self.base=types.SimpleNamespace(TensorAttr=lambda **kw:types.SimpleNamespace(**kw),
            query=lambda api,ctx,cmd,a:copy.deepcopy(api[cmd][a.index]),attr_record=lambda row:row)
        self.np=np;self.last_timings={}
        self.inputs=self._descriptors(1,self.input_shapes,names);self.outputs=self._descriptors(2,self.output_shapes,output_names)
    def run(self,values):
        assert all(x.shape==shape and x.dtype==np.float32 and x.flags.c_contiguous for x,shape in zip(values,self.input_shapes))
        self.received=[x.copy() for x in values];self.last_timings={'complete_boundary_seconds':0.}
        return [np.array(x.transpose(0,2,3,1) if x.ndim==4 and row['format']==1 else x,copy=True,order='C')
            for x,row in zip(self.seeds,self.outputs)]


def main():
    source=ROOT/'run_global0_split_abba.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==PIN
    spec=importlib.util.spec_from_file_location('_v114_generated_mixed_layout',source);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    Adapter=M.mixed_rank_context(TransportFixture);checks=[]
    global_values=np.arange(np.prod(GLOBAL),dtype=np.float32).reshape(GLOBAL)
    heads=np.arange(np.prod(BHSD),dtype=np.float32).reshape(BHSD)
    for fmt in (0,3):
        owner=Adapter([GLOBAL],['input'],[fmt],[GLOBAL],['output'],[fmt],[global_values])
        out=owner.run([global_values])[0];assert np.array_equal(out,global_values) and np.array_equal(owner.received[0],global_values)
        assert not np.shares_memory(out,global_values);checks.append(dict(case='complete_block_rank3_identity',format=fmt,passed=True))
    for output_formats in ([0,0,0],[1,0,1]):
        seeds=[heads+i for i in range(3)]
        owner=Adapter([GLOBAL],['input'],[3],[BHSD]*3,['query_rope','key_rope','value'],output_formats,seeds)
        ys=owner.run([global_values]);assert all(np.array_equal(y,x) and not np.shares_memory(y,x) for y,x in zip(ys,seeds))
        checks.append(dict(case='prefix_three_distinct_owned_BHSD_outputs',formats=output_formats,passed=True))
    for fmt in (0,1):
        owner=Adapter([GLOBAL,BHSD],['input','attention_values'],[3,fmt],[GLOBAL],['output'],[3],[global_values])
        y=owner.run([global_values,heads])[0];assert np.array_equal(y,global_values) and np.array_equal(owner.received[0],global_values)
        expected=heads.transpose(0,2,3,1) if fmt==1 else heads
        assert np.array_equal(owner.received[1],expected) and not np.shares_memory(owner.received[1],heads)
        if fmt==1:assert owner.received[1][0,2081,63,15]==heads[0,15,2081,63]
        checks.append(dict(case='suffix_mixed_rank_original_residual_and_attention_axes',attention_format=fmt,passed=True))
    for bad in ('rank3_format','rank4_shape','packed'):
        try:Adapter([GLOBAL],['input'],[3],[BHSD],['query_rope'],[1],[heads],bad)
        except ValueError:checks.append(dict(rejected=bad,passed=True))
        else:raise AssertionError('Unsafe mixed descriptor accepted: '+bad)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==PIN
    row=dict(schema='v114_generated_mixed_rank_layout_check',status='passed',runtime_source_sha256=PIN,
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,
        actual_SDK_NN_GPU_board_execution=False,attention_model_math_executed=False,runtime_source_unchanged=True)
    path=ROOT/'runtime-mixed-layout-check.json';path.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row));return 0


if __name__=='__main__':raise SystemExit(main())
