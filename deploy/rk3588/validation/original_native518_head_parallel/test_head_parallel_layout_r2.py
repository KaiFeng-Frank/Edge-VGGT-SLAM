#!/usr/bin/env python3
"""Generated distinct-axis logical-layout fixtures; no SDK/NN/GPU/board calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import types
import numpy as np

ROOT=Path(__file__).resolve().parent
PIN='e74ad2969551f90ef7d6f6eabebd3a1aa67bbb4dd819738f161a37189ee429f6'


def descriptor(name,index,heads,fmt):
    shape=[1,2082,64,heads] if fmt==1 else [1,heads,2082,64]
    return dict(name=name,index=index,shape=shape,n_elems=heads*2082*64,size=heads*2082*64*2,
        format=fmt,type=1,quantization_type=2,zero_point=0,scale=1.,pass_through=0)


class FixtureParent:
    """Typed-memory identity transport fixture, not an attention implementation."""
    def __init__(self,heads,output_format=1,bad=None):
        self.input_shapes=[(1,heads,2082,64)]*3;self.output_shapes=[(1,heads,2082,64)]
        input_rows=[descriptor(name,i,heads,1) for i,name in enumerate(('query_rope','key_rope','value'))]
        output_rows=[descriptor('attention_values',0,heads,output_format)]
        if bad=='shape':input_rows[0]['shape']=[1,64,2082,heads]
        if bad=='scale':input_rows[0]['scale']=2.
        if bad=='packed':input_rows[0]['format']=2
        if bad=='name':input_rows[0]['name']='changed_original_query'
        self.api={1:input_rows,2:output_rows};self.context=None
        self.base=types.SimpleNamespace(TensorAttr=lambda **args:types.SimpleNamespace(**args),
            query=lambda api,ctx,cmd,attr:copy.deepcopy(api[cmd][attr.index]),attr_record=lambda row:row)
        self.np=np;self.last_timings={}
        self.inputs=self._descriptors(1,self.input_shapes,('query_rope','key_rope','value'))
        self.outputs=self._descriptors(2,self.output_shapes,('attention_values',))
    def run(self,inputs):
        assert all(x.dtype==np.float32 and x.shape==shape and x.flags.c_contiguous for x,shape in zip(inputs,self.input_shapes))
        self.received=[x.copy() for x in inputs];self.last_timings={'complete_boundary_seconds':0.}
        value=inputs[0] if self.outputs[0]['format']==1 else inputs[0].transpose(0,3,1,2)
        return [np.array(value,copy=True,order='C')]


def main():
    path=ROOT/'run_sdpa_head_parallel_abba.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==PIN
    spec=importlib.util.spec_from_file_location('_v112_layout_fixture',path);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    Adapter=M.explicit_layout_context(FixtureParent);checks=[]
    for heads in (16,6,5):
        original=np.arange(heads*2082*64,dtype=np.float32).reshape(1,heads,2082,64)
        expected=original.transpose(0,2,3,1).copy()
        for fmt in (1,0):
            owner=Adapter(heads,fmt);output=owner.run([original,original.copy(),original.copy()])[0]
            assert np.array_equal(output,original) and not np.shares_memory(output,original)
            assert all(np.array_equal(x,expected) and not np.shares_memory(x,original) for x in owner.received)
            assert owner.received[0][0,2081,63,heads-1]==original[0,heads-1,2081,63]
            assert owner.input_shapes==[(1,2082,64,heads)]*3 and owner.semantic_input_shapes==[(1,heads,2082,64)]*3
            checks.append(dict(heads=heads,logical_output_format=fmt,all_distinct_axis_values_roundtrip_equal=True,
                owned_input_and_output_copies=True,full_original_shapes_preserved=True))
    for bad in ('shape','scale','packed','name'):
        try:Adapter(16,1,bad)
        except ValueError:checks.append(dict(rejected_descriptor=bad,passed=True))
        else:raise AssertionError('Unsafe descriptor accepted: '+bad)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==PIN
    record=dict(schema='v112_generated_logical_format_adapter_r2_checks',status='passed',runtime_sha256=PIN,
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,
        actual_SDK_NN_GPU_board_execution=False,attention_math_executed=False,runtime_source_unchanged=True)
    (ROOT/'runtime-layout-r2-check.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record));return 0


if __name__=='__main__':raise SystemExit(main())
