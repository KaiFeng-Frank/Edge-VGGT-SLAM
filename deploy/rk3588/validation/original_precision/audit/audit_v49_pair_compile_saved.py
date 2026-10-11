#!/usr/bin/env python3
"""Audit frozen original-VGGT saved PTQ records/logs only; never invoke a toolchain.

CLI paths locate private saved evidence; frozen sibling pure contracts are SHA pinned.
The complete output contains operator and allocation metadata, not weight/tensor values.
This parser never imports Torch/ORT/RKNN, calibrates, or connects to the board."""
import argparse
from collections import Counter
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
RUNTIME_CONTRACTS_SHA='44c0a6a1dcca66ee09b2f15bb0e2f0ddc7b7eb3f6cb44fdc3b576e9f75463bc2'
CAPTURE_CONTRACTS_SHA='6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8'
COMPILER_SHA='09c6e33be328f6f0c74265c3bdb2fa2c281569ffb76bbbd61db9cf509f81da24'
QUANT_CONTRACTS_SHA='47237b1a1e1614822c75180efd15be3e5c708b610f587f698fe9b37fcf1bc649'


def need(condition,message='Saved original pair compile contract mismatch'):
    if not condition:raise ValueError(message)


def pin(path,expected,pins):
    path=Path(path).resolve(strict=True)
    need(isinstance(expected,str) and re.fullmatch(r'[0-9a-f]{64}',expected) is not None)
    need(path.is_file() and sha(path)==expected,f'Saved record SHA mismatch: {path}')
    pins[str(path)]=expected
    return path


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-dir',type=Path,required=True)
    parser.add_argument('--compile-sha256',required=True)
    parser.add_argument('--capture-dir',type=Path,required=True)
    parser.add_argument('--capture-sha256',required=True)
    parser.add_argument('--toolchain-preflight',type=Path,required=True)
    parser.add_argument('--toolchain-preflight-sha256',required=True)
    parser.add_argument('--compiler-log',type=Path,required=True)
    parser.add_argument('--compiler-log-sha256',required=True)
    parser.add_argument('--verbose-log',type=Path,required=True)
    parser.add_argument('--verbose-log-sha256',required=True)
    parser.add_argument('--runtime-contracts',type=Path,default=HERE.parent/'runtime/v49_quantized_pair_runtime_contracts.py')
    parser.add_argument('--capture-contracts',type=Path,default=HERE.parent/'capture/v49_capture_contracts.py')
    parser.add_argument('--output',type=Path,required=True,help='New exclusive full private parse JSON; contains no tensor values')
    return parser.parse_args()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def source(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    args=arguments(); evidence={}
    C=source(pin(args.runtime_contracts,RUNTIME_CONTRACTS_SHA,evidence),'_v49_saved_compile_C')
    K=source(pin(args.capture_contracts,CAPTURE_CONTRACTS_SHA,evidence),'_v49_saved_compile_K')
    receipt=pin(args.compile_dir/'compile.json',args.compile_sha256,evidence)
    compiled=json.loads(receipt.read_text())
    need(not compiled.get('synthetic_result_not_evidence',False),'Synthetic receipt is not actual evidence')
    output=compiled.get('output',{})
    need(output.get('file')=='pair-int8.rknn')
    model=pin(args.compile_dir/output['file'],output.get('sha256'),evidence)
    model_sha=sha(model)
    need(model.stat().st_size==output.get('size'))
    C.check_quantized_compile(compiled,COMPILER_SHA,args.capture_sha256,model_sha,
        QUANT_CONTRACTS_SHA,CAPTURE_CONTRACTS_SHA)
    captured,samples,capture_pins=K.load_capture(args.capture_dir,args.capture_sha256)
    evidence.update(capture_pins)
    evals=C.bind_compilation_to_capture(compiled,captured,samples)
    dataset=pin(args.compile_dir/'dataset.txt',compiled['dataset']['sha256'],evidence)
    content=''.join(row['path']+'\n' for row in compiled['dataset']['rows']).encode()
    need(dataset.read_bytes()==content and sha(dataset)==compiled['dataset']['sha256'])
    need(dataset.stat().st_size==compiled['dataset']['bytes'])
    preflight=pin(args.toolchain_preflight,args.toolchain_preflight_sha256,evidence)
    imports=json.loads(preflight.read_text())
    need(imports.get('status')=='imports_passed_no_model_execution')
    need(all(imports.get(name) is False for name in ('model_loaded','model_forward_executed','RKNN_object_created','ORT_session_created')))
    need(imports['prefix']==compiled['toolchain_prefix'] and imports['machine']==compiled['machine'])
    modules={}
    for name,path in compiled['module_source_paths'].items():
        need(Path(path).is_relative_to(compiled['toolchain_prefix']))
        if name in imports['modules']:
            actual=imports['modules'][name]
            need(actual['path']==path and compiled['source_and_input_hashes'][path]==actual['sha256'])
            modules[name]={'pinned_board_source_matches_import_preflight':True,
                           'version':compiled['actual_module_versions'][name]}
        else:
            modules[name]={'board_wrapper_source_pinned_in_receipt':True,'path':path,
                           'sha256':compiled['source_and_input_hashes'][path]}
    need(compiled['actual_module_versions']=={'torch':'2.2.0','numpy':'1.26.4','onnx':'1.16.1'})
    compiler_log=pin(args.compiler_log,args.compiler_log_sha256,evidence)
    verbose_log=pin(args.verbose_log,args.verbose_log_sha256,evidence)
    log=re.sub(r'\x1b\[[0-9;]*m','',compiler_log.read_text())
    need(compiled['status'] in log and sha(receipt) in log)
    need('The config.mean_values is None, zeros will be set for input 0!' in log)
    need('The config.std_values is None, ones will be set for input 0!' in log)
    prefix=re.compile(r'^D RKNN:\s*\[[^\]]+\]\s*')
    op=re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.+?)\s+(\d+/\d+/\d+)\s+(\d+)\s+(.+?)\s*$')
    weight=re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.+?)\s*\|\s*(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s*$')
    ops,weights,warnings,section=[],[],[],None
    for line in log.splitlines():
        if line.startswith('W '):warnings.append(line)
        if 'ID   OpType' in line and 'Target' in line:section='ops';continue
        if 'ID  User' in line and 'Tensor' in line and 'NativeShape' in line:section='native';continue
        if 'ID  User' in line and 'Tensor' in line and 'OrigShape' in line and 'NativeShape' not in line:section='weights';continue
        stripped=prefix.sub('',line)
        if stripped==line:continue
        if section=='ops':
            match=op.match(stripped)
            if match:
                ops.append({'id':int(match[1]),'op':match[2],'data_type_token':match[3],'target':match[4],
                            'input_and_output_shapes_text':match[5],
                            'DDR_NPU_total_cycles':[int(v) for v in match[6].split('/')],
                            'RW_KB':int(match[7]),'name':match[8]})
            elif re.match(r'^\d+\s',stripped):raise ValueError('Unparsed compile operator '+line)
        elif section=='weights':
            match=weight.match(stripped)
            if match:
                start,end,size=[int(match[i],16) for i in (6,7,8)]
                need(end-start==size)
                shape=ast.literal_eval(match[5]);elements=1
                if type(shape) is int:shape=(shape,)
                for value in shape:
                    need(type(value) is int and value>0)
                    elements*=value
                weights.append({'operator_id':int(match[1]),'op':match[2],'tensor':match[3],
                                'data_type_token':match[4],'shape':list(shape),'start':start,'end':end,
                                'allocated_bytes':size,'allocated_bytes_per_logical_element':size/elements,
                                'allocated_extent_may_include_native_padding':True})
            elif re.match(r'^\d+\s',stripped):raise ValueError('Unparsed compile weight '+line)
    need([row['id'] for row in ops]==list(range(54)))
    need(ops[0]['op']=='InputOperator' and ops[-1]['op']=='OutputOperator')
    internal=ops[1:-1];cpu=[row for row in internal if row['target']=='CPU']
    need(len(cpu)==1 and cpu[0]['id']==24 and cpu[0]['name']=='Reshape:frame0/value_tp_rs')
    need(Counter(row['target'] for row in internal)=={'NPU':51,'CPU':1})
    need(Counter(row['data_type_token'] for row in ops)=={'INT8':54})
    need(len(weights)==57 and Counter(row['data_type_token'] for row in weights)=={'INT8':18,'INT64':14,'INT32':9,'FLOAT16':8,'FLOAT':8})
    need([row for row in warnings if 'height*width' in row and 'Transpose' in row] or
         len([row for row in warnings if '16384' in row])==4,'Expected recorded transpose fallback warnings')
    need(all(row['allocated_bytes_per_logical_element']==4 for row in weights if row['data_type_token']=='FLOAT'))
    need(all(row['allocated_bytes_per_logical_element']==2 for row in weights if row['data_type_token']=='FLOAT16'))
    report={'schema':'v49_original_VGGT_board_PTQ_saved_compile_independent_audit',
        'status':'passed_actual_saved_compile_contract_no_runtime_adoption','script_sha256':sha(__file__),
        'actual_model_ORT_SDK_compiler_calibration_executed_by_auditor':False,'board_access_by_auditor':False,
        'actual_compile_receipt_sha256':sha(receipt),'actual_model_sha256':model_sha,'actual_model_bytes':model.stat().st_size,
        'actual_main_log_sha256':sha(compiler_log),'actual_verbose_log_sha256':sha(verbose_log),
        'toolchain_import_preflight_sha256':sha(preflight),
        'runtime_contracts_sha256':RUNTIME_CONTRACTS_SHA,'capture_contracts_sha256':CAPTURE_CONTRACTS_SHA,
        'capture_manifest_sha256':compiled['capture_manifest_sha256'],'saved_capture_files_checked':len(capture_pins),
        'actual_dataset_sha256':sha(dataset),'dataset_rows':compiled['dataset']['rows'],
        'evaluation_sample_ids_excluded':[s['sample_id'] for s in evals],
        'calibration_rows_exactly_match_original_saved_capture':True,
        'configuration':compiled['configuration'],'input_transform':compiled['input_transform'],'build_options':compiled['build_options'],
        'actual_compiler_operations':compiled['operations'],'module_source_checks':modules,
        'release_code':compiled['release_code'],'released':compiled['released'],
        'source_and_inputs_unchanged':compiled['source_and_inputs_unchanged'],
        'cwd_restored':compiled['cwd_restored'],'environment_restored':compiled['environment_restored'],'warnings':warnings,
        'final_compile_operator_table':ops,'compiled_internal_target_counts':dict(Counter(r['target'] for r in internal)),
        'compiled_operator_data_type_token_counts':dict(Counter(r['data_type_token'] for r in ops)),
        'compiled_internal_CPU_operators':cpu,'compiled_weight_and_constant_records':weights,
        'compiled_weight_constant_data_type_token_counts':dict(Counter(r['data_type_token'] for r in weights)),
        'Norm_FLOAT_bias_has_4_bytes_per_element':True,'Norm_FLOAT16_weight_has_2_bytes_per_element':True,
        'model_bytes_reduction_from_frozen_nonquant_percent':100*(55280948-model.stat().st_size)/55280948,
        'candidate_accepted':False,'runtime_operator_placement_or_accuracy_or_speed_validated_by_this_receipt':False,
        'limitations':['Verifies saved actual root board compilation records; auditor ran no toolchain',
            'Final compiler table has one internal CPU Reshape; four early Transpose fallback warnings are not the final placement',
            'Operator IO tags say INT8, while constants include FLOAT/FLOAT16 Norm, INT8 linear/RoPE, INT32 bias and INT64 shape',
            'Tags/storage do not prove every arithmetic instruction precision; runtime operator profiling is still required',
            'Compiler size reduction is for this original two-block component, not full checkpoint or SLAM memory',
            'Heldout FP32, full SLAM/trajectory, real-time and stability gates remain separate'],
        'root_execution_order_update':'PTQ compilation completed before new heldout ORT; root runs heldout ORT before SDK A/B'}
    after={path:sha(path) for path in evidence}
    need(after==evidence,'Saved data/source changed during audit')
    report['saved_sources_and_inputs_unchanged']=True
    report['input_saved_artifact_paths']={name:str(getattr(args,name)) for name in
        ('compile_dir','capture_dir','toolchain_preflight','compiler_log','verbose_log')}
    output=args.output
    with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'receipt_sha256':sha(output),'operator_rows':len(ops),
        'weight_records':len(weights),'internal_targets':report['compiled_internal_target_counts'],
        'weight_types':report['compiled_weight_constant_data_type_token_counts'],
        'model_size_reduction_percent':report['model_bytes_reduction_from_frozen_nonquant_percent']}))


if __name__=='__main__':main()
