#!/usr/bin/env python3
"""Audit saved original-VGGT SDK operator tables only; never run an executor."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

ROW = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.+?)\s+'
                 r'(\d+/\d+/\d+)\s+(\d+)\s+(.*?)\s+'
                 r'(\d+(?:\.\d+)?%/\d+(?:\.\d+)?%/\d+(?:\.\d+)?%)\s+(\d+)\s+(.+?)\s*$')
TOTAL = re.compile(r'Total Operator Elapsed Per Frame Time\(us\):\s*(\d+)')
ROLES = ('nonquant_fp16','ptq_w8a8')
RUNTIME_SHA = '33fc5deb749c8b3ceaa47271e389f5fbbfc2dba6db93aa19887b2fc41e0e7882'
CONTRACTS_SHA = '44c0a6a1dcca66ee09b2f15bb0e2f0ddc7b7eb3f6cb44fdc3b576e9f75463bc2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(raw):
    """Every numeric operator row must parse; dtype tokens remain verbatim."""
    rows=[]
    for line in raw.splitlines():
        if not re.match(r'^\s*\d+\s+',line):
            continue
        match=ROW.match(line)
        if not match:
            raise ValueError(f'Unparsed SDK operator row: {line}')
        rows.append({'id':int(match[1]),'op':match[2],'data_type_token':match[3],
                     'target':match[4],'input_and_output_shapes_text':match[5],
                     'DDR_NPU_total_cycles':[int(v) for v in match[6].split('/')],
                     'microseconds':int(match[7]),'MAC_usage_text':match[8],
                     'workload_0_1_2_text':match[9],'RW_KB':int(match[10]),'name':match[11]})
    totals=TOTAL.findall(raw)
    if (len(totals)!=1 or not rows or [r['id'] for r in rows]!=list(range(1,len(rows)+1))
            or sum(r['microseconds'] for r in rows)!=int(totals[0])):
        raise ValueError('Incomplete SDK operator table or inconsistent exact time sum')
    boundary=[r for r in rows if r['op'] in ('InputOperator','OutputOperator')]
    if (len(boundary)!=2 or rows[0]['op']!='InputOperator' or rows[-1]['op']!='OutputOperator'):
        raise ValueError('Expected exactly one logical input/output boundary in paired graph profile')
    return rows,int(totals[0])


def identity(row):
    return tuple(row[k] for k in ('id','op','data_type_token','target','name','input_and_output_shapes_text'))


def lifecycle(report,normal):
    if (report.get('schema')!='v49_original_VGGT_pair_precision_SDK_AB'
            or not report.get('status','').startswith('completed_')
            or report.get('script_sha256')!=RUNTIME_SHA or report.get('contracts_sha256')!=CONTRACTS_SHA
            or report.get('normal_benchmark') is not normal
            or report.get('perf_detail_collection_enabled') is normal
            or report.get('successful_synchronous_call_count')!=16 or report.get('npu_run_api_call_count')!=16
            or len(report.get('calls',[]))!=16 or len(report.get('sdk_calls',[]))!=16
            or len(report.get('paired_implementation_differences',[]))!=8
            or report.get('candidate_accepted') is not False
            or report.get('inference_on_board_only') is not True
            or report.get('source_and_inputs_unchanged') is not True
            or report.get('affinity_restored') is not True or report.get('environment_restored') is not True
            or report.get('cleanup_errors')!=[]
            or report.get('rtol')!=1e-4 or report.get('atol')!=1e-5):
        raise ValueError('Completed clean source-pinned sixteen-call original precision receipt required')
    contexts=report.get('context_records',{})
    if (set(contexts)!=set(ROLES) or any(r.get('context_destroyed') is not True
            or r.get('init_flags')!=(0 if normal else 8) for r in contexts.values())):
        raise ValueError('Fresh contexts and matching normal/profile flags required')
    samples=report['evaluation_sample_ids']
    if len(samples)!=2 or len(set(samples))!=2:
        raise ValueError('Two explicit heldout IDs required')
    for sample in samples:
        for role in ROLES:
            calls=[r for r in report['calls'] if r['sample_id']==sample and r['role']==role]
            if [r['sample_call_index'] for r in calls]!=list(range(4)):
                raise ValueError('Expected four calls per heldout input/implementation')
    return samples


def summarize_tables(tables):
    all_rows=tables[0]['rows']
    baseline=[identity(r) for r in all_rows]
    if any([identity(r) for r in t['rows']]!=baseline for t in tables):
        raise ValueError('Operator type/target/name/shape changed between same-implementation profiles')
    repeats=[t for t in tables if t['sample_call_index']>0]
    if len(tables)!=8 or len(repeats)!=6:
        raise ValueError('Exactly eight profiles / six repeats per implementation required')
    internal=all_rows[1:-1]
    groups=Counter((r['op'],r['data_type_token'],r['target']) for r in all_rows)
    families=[{'op':key[0],'data_type_token':key[1],'target':key[2],'count_per_call':count,
               'mean_all_eight_microseconds':statistics.mean(sum(r['microseconds'] for r in t['rows']
                           if (r['op'],r['data_type_token'],r['target'])==key) for t in tables)}
              for key,count in sorted(groups.items())]
    operators=[{**row,'mean_all_eight_microseconds':statistics.mean(t['rows'][i]['microseconds'] for t in tables),
                      'mean_six_repeat_microseconds':statistics.mean(t['rows'][i]['microseconds'] for t in repeats)}
               for i,row in enumerate(all_rows)]
    result={'operator_rows_per_call':len(all_rows),'internal_rows_per_call':len(internal),
            'internal_target_counts':dict(Counter(r['target'] for r in internal)),
            'boundary_target_counts':dict(Counter(r['target'] for r in (all_rows[0],all_rows[-1]))),
            'internal_data_type_token_counts':dict(Counter(r['data_type_token'] for r in internal)),
            'all_listed_internal_targets_NPU':all(r['target']=='NPU' for r in internal),
            'all_listed_internal_data_type_tokens_INT8':all(r['data_type_token']=='INT8' for r in internal),
            'whole_original_model_INT8_or_NPU_proven':False,
            'all_eight_operator_mean_microseconds':statistics.mean(t['exact_table_total_microseconds'] for t in tables),
            'six_repeat_operator_mean_microseconds':statistics.mean(t['exact_table_total_microseconds'] for t in repeats),
            'profile_operator_families':families,'profile_named_operators':operators,
            'original_subregions':{}}
    for role in ('frame0','global0'):
        result['original_subregions'][role]={}
        for region in ('MLP','SDPA','reshape_transpose'):
            def selected(row):
                if role+'/' not in row['name']:return False
                if region=='MLP':return 'mlp.fc1' in row['name'] or 'mlp.fc2' in row['name']
                if region=='SDPA':return 'SDPAttention' in row['op'] or 'attention_values' in row['name']
                return row['op'] in ('Reshape','Transpose')
            subset=[r for r in all_rows if selected(r)]
            result['original_subregions'][role][region]={
                'operator_ids':[r['id'] for r in subset],
                'data_type_tokens':dict(Counter(r['data_type_token'] for r in subset)),
                'targets':dict(Counter(r['target'] for r in subset)),
                'mean_all_eight_microseconds':statistics.mean(sum(r['microseconds'] for r in t['rows'] if selected(r)) for t in tables)}
    return result


def audit(profile_dir,profile_sha,normal_dir=None,normal_sha=None):
    profile_dir=Path(profile_dir)
    if sha(profile_dir/'run.json')!=profile_sha:raise ValueError('Exact saved profile receipt SHA required')
    profile=json.loads((profile_dir/'run.json').read_text())
    samples=lifecycle(profile,False)
    tables={role:[] for role in ROLES}
    for call in profile['calls']:
        detail=call['separate_profile_detail']
        relative=Path(detail['file'])
        path=(profile_dir/relative).resolve(strict=True)
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(profile_dir.resolve()):
            raise ValueError('Profile detail path must be receipt-owned')
        raw=path.read_bytes()
        if (len(raw)!=detail['bytes'] or hashlib.sha256(raw).hexdigest()!=detail['raw_sha256']
                or sha(path)!=detail['saved_text_sha256']):
            raise ValueError('Raw SDK operator bytes/hash differ from saved detail')
        rows,total=parse(raw.decode('utf-8',errors='strict'))
        tables[call['role']].append({'sample_id':call['sample_id'],'sample_call_index':call['sample_call_index'],
                                    'sdk_call_index':call['sdk_call_index'],'detail_file':str(relative),
                                    'detail_sha256':sha(path),'exact_table_total_microseconds':total,'rows':rows})
    result={'schema':'v49_original_VGGT_saved_profile_independent_audit','status':'saved_tables_checked',
            'script_sha256':sha(__file__),'actual_model_OR_ORT_OR_SDK_OR_compiler_executed_by_auditor':False,
            'board_access_by_auditor':False,'profile_receipt_sha256':profile_sha,
            'implementations':{role:summarize_tables(tables[role]) for role in ROLES},
            'all_call_tables':tables,'candidate_accepted':False,
            'limitations':['SDK table DataType tokens are reported verbatim; do not infer every arithmetic instruction precision',
                           'Internal placements cover only the two original graph blocks, never whole VGGT/SLAM',
                           'Profile operator times and normal SDK IO come from separate calls; never sum them',
                           'Two heldout inputs/four calls each do not establish trajectory, P95, camera or fresh-pose performance']}
    if normal_dir is not None:
        normal_dir=Path(normal_dir)
        if sha(normal_dir/'run.json')!=normal_sha:raise ValueError('Exact saved normal receipt SHA required')
        normal=json.loads((normal_dir/'run.json').read_text())
        if lifecycle(normal,True)!=samples:raise ValueError('Normal/profile heldout inputs differ')
        for key in ('capture_manifest_sha256','ptq_compile_sha256','ptq_model_sha256','nonquant_model_sha256','pair_graph_sha256'):
            if profile[key]!=normal[key]:raise ValueError('Normal/profile source/model provenance differs')
        result['normal_receipt_sha256']=normal_sha
        result['separate_normal_SDK_IO']={}
        for role in ROLES:
            calls=[r for r in normal['calls'] if r['role']==role and r['sample_call_index']>0]
            if len(calls)!=6:raise ValueError('Actual normal IO mean needs six repeats')
            mean=statistics.mean(r['io_total_seconds'] for r in calls)
            saved=normal['summaries']['implementations'][role]
            if abs(saved['two_input_subsequent_io_mean_seconds']-mean)>1e-12 or saved['mean_denominator']!=6:
                raise ValueError('Saved normal timing denominator/mean inconsistent')
            result['separate_normal_SDK_IO'][role]={'repeat_call_count':6,'repeat_mean_seconds':mean,
                                                  'strict_original_FP32_gate_passed':saved['strict_original_FP32_gate_passed']}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-dir',type=Path,required=True)
    parser.add_argument('--profile-sha256',required=True)
    parser.add_argument('--normal-dir',type=Path)
    parser.add_argument('--normal-sha256')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if (args.normal_dir is None)!=(args.normal_sha256 is None):parser.error('Normal directory/SHA must be provided together')
    report=audit(args.profile_dir,args.profile_sha256,args.normal_dir,args.normal_sha256)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'output_sha256':sha(args.output)}))


if __name__=='__main__':main()
