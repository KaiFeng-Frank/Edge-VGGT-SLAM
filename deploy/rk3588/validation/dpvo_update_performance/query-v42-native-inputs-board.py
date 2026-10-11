#!/usr/bin/env python3
"""Board-owned nine-shape input query only; no input submission or forward.

The frozen v12 ABI supplies structures, not its inference binding or executor.
Command1 is the ordinary logical input attr; native commands8/10 are separate.
DPVO research, Princeton University (2022), MIT; SDK/header are not distributed.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

INDEX_SHA256 = 'eeb9241c9eab52039aa64fcf6546ce4624d1388756a9e919f42c22de4083a85c'
ABI_SHA256 = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA256 = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA256 = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
MODEL_KEYS = frozenset([f'B{b}_D{d}_M16' for b in (1,2) for d in range(4,8)]+['B1_D3_M16'])
COMPILERS = dict(v40='3ccd9002247bf3ff15e87bf04db2e3462db782c6d5a637f50b98094bcd584294',
                 v41='4ddab438d294172380876d82ae1e6fd7a4f52ed797f901248fec492d331f2ff1',
                 v42='8575fd63ca004f59c494a01e443e1bf76b7d0c9196a185b1a7b641e1dd900f53')
SCHEMAS = dict(v40='v40_M16_grouped_Update_RKNN_compilation_diagnostic',
               v41='v41_online_M16_Update_RKNN_compilation_diagnostic',
               v42='v42_online_M16_EOF_Update_RKNN_compilation_diagnostic')
ATTR_COMMANDS = ((1,'logical_input'),(8,'native_input_NC1HWC2_query'),(10,'native_NHWC_input_query'))
CONFIGURATION = dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False)
SUCCESS = 'queried_all_nine_shapes_inputs_not_inference_or_transport_validated'


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): digest.update(block)
    return digest.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and isinstance(expected,str) and len(expected)==64
            and all(c in '0123456789abcdef' for c in expected) and sha(path)==expected,
            'Pinned SHA differs: '+path.name)
    require(str(path) not in pins or pins[str(path)]==expected, 'Conflicting source pin')
    pins[str(path)] = expected
    return path


def load_abi(path):
    spec = importlib.util.spec_from_file_location('_v42_native_query_frozen_ABI',path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def bind_query_api(library, abi):
    signatures = dict(rknn_init=[ct.POINTER(abi.CONTEXT),ct.c_void_p,ct.c_uint32,ct.c_uint32,ct.c_void_p],
        rknn_set_core_mask=[abi.CONTEXT,ct.c_int],
        rknn_query=[abi.CONTEXT,ct.c_int,ct.c_void_p,ct.c_uint32],rknn_destroy=[abi.CONTEXT])
    for name, args in signatures.items():
        function = getattr(library,name);function.argtypes=args;function.restype=ct.c_int
    return library


def validate_index(value):
    require(value['schema']=='v42_online_M16_Update_compile_index'
        and value['status']=='staged_nine_online_shapes_diagnostic'
        and value['candidate_accepted'] is False and set(value['models'])==MODEL_KEYS,
        'Exact nine-shape diagnostic index required')
    require(value['old_online_compile_index_sha256']=='8c3f1c3b92e14a6742a984f750a5eb4c1bbf6397ca31440b96b0768333dc21fa'
        and value['parent_manifest_sha256']=='5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
        and value['additions_manifest_sha256']=='2f5b3dd89b5863963bdfa52ffa213af02f08b6654a7012f5980ea3189d7ba47f'
        and value['eof_manifest_sha256']=='6837252e28ee109fc67680ed197825e403ec727ce3e7ad391a3090ea85ff234f',
        'Frozen index ancestry differs')


def load_models(index, pins):
    result = []
    for key,row in sorted(index['models'].items()):
        lineage = 'v42' if key=='B1_D3_M16' else ('v41' if key in ('B1_D4_M16','B1_D6_M16') else 'v40')
        require(row['lineage']==lineage, 'Shape lineage differs')
        directory = Path(row['compile_dir']).resolve(strict=True)
        compiled_path = pin(directory/'compile.json',row['compile_sha256'],pins)
        compiled = json.loads(compiled_path.read_text())
        require(compiled['schema']==SCHEMAS[lineage] and compiled['script_sha256']==COMPILERS[lineage]
            and compiled['status']=='compiled_group_diagnostic_not_runtime_validated'
            and compiled['model_key']==key and compiled['configuration']==CONFIGURATION
            and compiled['output']==row['model'] and compiled['released'] is True
            and compiled['sources_and_inputs_unchanged'] is True and compiled['candidate_accepted'] is False,
            'Compiled shape/artifact/diagnostic contract differs')
        b,d=int(key.split('_')[0][1:]),int(key.split('_')[1][1:])
        ins=[dict(name=n,dtype='float32',shape=[b,c,d,16]) for n,c in (('net',384),('inp',384),('corr',882))]
        outs=[dict(name=n,dtype='float32',shape=[b,c,d,16]) for n,c in (('net_out',384),('delta',2),('weight',2))]
        require(compiled['logical_inputs']==ins and compiled['logical_outputs']==outs
            and [v['name'] for v in compiled['operations']]==['config','load_onnx','build','export_rknn']
            and all(type(v['code']) is int and v['code']==0 for v in compiled['operations']),
            'Original complete-source IO or compilation operations differ')
        artifact=row['model'];require(Path(artifact['file']).name==artifact['file']
            and type(artifact['bytes']) is int and 0<artifact['bytes']<=128<<20,'Bounded model basename/size required')
        model=pin(directory/artifact['file'],artifact['sha256'],pins)
        require(model.stat().st_size==artifact['bytes'],'Actual model bytes differ')
        result.append(dict(model_key=key,model_path=model,model=artifact,
            compile_sha256=row['compile_sha256'],logical_inputs=ins,logical_outputs=outs,
            export_strict_FP32_gate_passed=compiled['export_strict_FP32_gate_passed']))
    return result


def full_attr_record(attr):
    rank=int(attr.n_dims)
    scale=float(attr.scale)
    return dict(index=int(attr.index),n_dims=rank,dims=list(attr.dims),
        shape=list(attr.dims[:rank]) if 1<=rank<=16 else None,
        name=bytes(attr.name).decode('utf-8',errors='replace'),name_bytes_hex=bytes(attr.name).hex(),
        n_elems=int(attr.n_elems),size=int(attr.size),fmt=int(attr.fmt),type=int(attr.type),
        qnt_type=int(attr.qnt_type),fl=int(attr.fl),zp=int(attr.zp),
        scale=scale if math.isfinite(scale) else repr(scale),scale_finite=math.isfinite(scale),
        w_stride=int(attr.w_stride),h_stride=int(attr.h_stride),
        size_with_stride=int(attr.size_with_stride),pass_through=int(attr.pass_through))


def probe_context(api, abi, payload, row, keepalive):
    """Init/query/destroy one context; retain host buffer if destroy fails."""
    context=abi.CONTEXT();buffer=ct.create_string_buffer(payload)
    row.update(initialization_attempted=True,initialized=False,context_destroyed=None,
        model_host_buffer_bytes=len(buffer),core_mask_requested=7,init_flags=0,
        queries=[],cleanup_errors=[],sdk_run_calls=0,model_inference_executed=False,NPU_used=False)
    try:
        code=int(api.rknn_init(ct.byref(context),ct.cast(buffer,ct.c_void_p),len(payload),0,None))
        row['init_code']=code;abi.check_call(code,'rknn_init')
        require(context.value!=0,'SDK returned a zero initialized context');row['initialized']=True
        code=int(api.rknn_set_core_mask(context,7));row['core_mask_code']=code;abi.check_call(code,'rknn_set_core_mask')
        version=abi.query(api,context,5,abi.SDKVersion())
        row['SDK_version']=bytes(version.api_version).decode('utf-8',errors='replace')
        row['driver_version']=bytes(version.drv_version).decode('utf-8',errors='replace')
        require(row['SDK_version'].startswith('2.3.2'),'Pinned SDK version differs')
        counts=abi.query(api,context,0,abi.InputOutputNum())
        row['operand_counts']=dict(inputs=int(counts.n_input),outputs=int(counts.n_output))
        require(counts.n_input==counts.n_output==3,'Exact three-input/output model required')
        for command,label in ATTR_COMMANDS:
            for index in range(3):
                attr=abi.TensorAttr(index=index)
                code=int(api.rknn_query(context,command,ct.byref(attr),ct.sizeof(attr)))
                row['queries'].append(dict(command=command,kind=label,input_index=index,
                    return_code=code,query_succeeded=code==0,attribute=full_attr_record(attr)))
        memory=abi.query(api,context,6,abi.MemSize())
        row['memory']={name:(list(getattr(memory,name)) if name=='reserved' else int(getattr(memory,name)))
                       for name,_ in abi.MemSize._fields_}
        row['all_input_attr_queries_succeeded']=all(v['query_succeeded'] for v in row['queries'])
        row['status']='queried' if row['all_input_attr_queries_succeeded'] else 'queried_with_failed_native_or_logical_commands'
    except BaseException as error:
        row.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if context.value:
            try:
                code=int(api.rknn_destroy(context));row['destroy_code']=code;row['context_destroyed']=code==0
                if code:row['cleanup_errors'].append('rknn_destroy returned '+str(code))
            except BaseException as error:
                row['context_destroyed']=False;row['cleanup_errors'].append(repr(error))
        if context.value and row['context_destroyed'] is not True:
            keepalive.append(buffer);row['model_host_buffer_retained_after_failed_destroy']=len(buffer)
            row['status']='failed'
        else:row['model_host_buffer_retained_after_failed_destroy']=0
    return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('index','abi-helper','header','library','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--index-sha256',required=True)
    p.add_argument('--diagnostic-allow-fp32-rejected',action='store_true')
    args=p.parse_args(argv)
    require(args.diagnostic_allow_fp32_rejected is True,'Explicit FP32-rejected diagnostic required')
    require(args.index_sha256==INDEX_SHA256,'Fixed nine-shape index SHA required')
    pins={};abi=load_abi(pin(args.abi_helper,ABI_SHA256,pins));abi.require_board()
    output=args.output_dir.resolve();require(not output.exists(),'New isolated output directory required')
    sources=[v.resolve() for k,v in vars(args).items() if isinstance(v,Path) and k!='output_dir']
    require(not any(output==v or output.is_relative_to(v) or v.is_relative_to(output) for v in sources),'Output/source overlap')
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='v42_nine_shape_logical_and_native_input_query',status='failed',
        script_sha256=sha(__file__),index_sha256=INDEX_SHA256,candidate_accepted=False,
        diagnostic_allow_fp32_rejected=True,historical_SDK_strict_FP32_rejected=True,
        SDK_FP32_equivalence_validated=False,actual_board_execution=True,
        model_inference_executed=False,sdk_run_calls=0,NPU_used=False,transport_equivalence_validated=False,
        native_layout_inferred_from_logical_attrs=False,query_commands=[5,0,1,8,10,6],
        bound_functions=['rknn_init','rknn_set_core_mask','rknn_query','rknn_destroy'],
        models=[],cleanup_errors=[],source_pins=pins,source_scope='Exact index/ABI/header/library plus all nine compile/model bytes; no weights/input arrays or mathematical executor loaded.')
    keepalive=[]
    try:
        pin(__file__,report['script_sha256'],pins)
        pin(args.header,HEADER_SHA256,pins);library=pin(args.library,LIBRARY_SHA256,pins)
        index=json.loads(pin(args.index,INDEX_SHA256,pins).read_text());validate_index(index)
        models=load_models(index,pins)
        report['abi']=abi.abi_record()
        require(report['abi']['context_size']==report['abi']['pointer_size']==8,'Pinned LP64 ABI required')
        require(not any(output==Path(v) or Path(v).is_relative_to(output) for v in pins),'Output/artifact overlap')
        api=bind_query_api(ct.CDLL(str(library)),abi)
        for model in models:
            path=model['model_path'];require(sha(path)==model['model']['sha256'],'Model changed before initialization')
            row={k:v for k,v in model.items() if k!='model_path'};report['models'].append(row)
            payload=path.read_bytes();require(len(payload)==model['model']['bytes'],'Model size changed before initialization')
            probe_context(api,abi,payload,row,keepalive);del payload
            if row['cleanup_errors']:report['cleanup_errors'].extend(row['cleanup_errors']);break
            if row['status']=='failed':break
        if len(report['models'])==9 and all(v['status']=='queried' and v['context_destroyed'] is True for v in report['models']):
            report['status']=SUCCESS
        elif (len(report['models'])==9 and not report['cleanup_errors']
                and all(v['status'] in ('queried','queried_with_failed_native_or_logical_commands')
                        and v['context_destroyed'] is True for v in report['models'])):
            report['status']='queried_nine_shapes_with_failed_commands_not_transport_validated'
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        report['source_hashes_after']={path:sha(path) if Path(path).is_file() else None for path in pins}
        report['sources_and_inputs_unchanged']=report['source_hashes_after']==pins
        report['contexts_initialized']=sum(v.get('initialized') is True for v in report['models'])
        report['contexts_destroyed']=sum(v.get('context_destroyed') is True for v in report['models'])
        report['model_host_buffers_retained_after_failed_destroy']=sum(len(v) for v in keepalive)
        if report['cleanup_errors'] or not report['sources_and_inputs_unchanged']:report['status']='failed'
        (output/'query.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','contexts_initialized','contexts_destroyed','sdk_run_calls')}),flush=True)
    return 0 if report['status']==SUCCESS else 1


if __name__=='__main__':raise SystemExit(main())
