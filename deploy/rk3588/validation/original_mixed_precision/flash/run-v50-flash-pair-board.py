#!/usr/bin/env python3
"""Same-toolchain Flash Attention off/on original-pair board diagnostic.

Reuse the frozen v49 typed SDK call/ownership implementation. No actual model
or SDK invocation is possible before the RK3588 identity and provenance gates.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
PARENT_SHA = '33fc5deb749c8b3ceaa47271e389f5fbbfc2dba6db93aa19887b2fc41e0e7882'
CONTRACT_SHA = '44c0a6a1dcca66ee09b2f15bb0e2f0ddc7b7eb3f6cb44fdc3b576e9f75463bc2'
CAPTURE_SHA = 'bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc'
CAPTURE_SOURCE_SHA = '6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8'
COMPILER_SHA = '709544077448718fe866e8e288810ac43110fb2afc98bd3a39cde5d6d3a965b0'
ROLES = ('ordinary', 'flash')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['parent', 'contracts', 'compiler-source', 'compile-dir', 'capture-source',
                 'capture-dir', 'runtime-helper', 'sdk-header', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--compile-sha256', required=True)
    parser.add_argument('--collect-perf-detail', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        parser.error('Actual execution runs only on RK3588S')
    import hashlib, importlib.util
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def load(path, expected):
        if digest(path) != expected: raise ValueError('Frozen source SHA mismatch')
        spec = importlib.util.spec_from_file_location('_v50_' + path.stem, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    C = load(args.contracts, CONTRACT_SHA)
    pins = {str(Path(__file__).resolve()): digest(__file__)}
    C.pin(args.contracts, CONTRACT_SHA, pins)
    R = C.source(args.runtime_helper, C.PARENT_RUNTIME_SHA, pins)
    board = R.require_board()
    output = args.output_dir.resolve()
    if Path.cwd() != BASE or output == BASE / 'experiments' or not output.is_relative_to(BASE / 'experiments'):
        parser.error('Project cwd and new experiment output directory required')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v50_original_pair_flash_SDK_AB', 'status': 'failed',
              'script_sha256': digest(__file__), 'board': board, 'machine': platform.machine(),
              'parent_source_sha256': PARENT_SHA, 'compile_sha256': args.compile_sha256,
              'original_graph_sha256': C.PAIR_GRAPH_SHA, 'capture_sha256': CAPTURE_SHA,
              'candidate_accepted': False, 'full_model_run': False, 'real_time_validated': False,
              'whole_model_quality_validated': False, 'strict_gate': {'rtol': C.RTOL, 'atol': C.ATOL},
              'normal_benchmark': not args.collect_perf_detail,
              'profile_separate_fresh_contexts': args.collect_perf_detail,
              'requested_sdk_call_count': 16, 'unmeasured_warmup_calls': 0,
              'contexts_live_together': True, 'calls': [], 'sdk_calls': [], 'context_records': {},
              'cleanup_errors': [], 'limits': [
                  'Complete original frame0/global0 pair only; no fresh pose/full SLAM claim',
                  'Two evaluation inputs excluded from calibration but reused for development; not a new blind quality test',
                  'Profile operator clocks and normal SDK IO have separate contexts and denominators',
                  'Original FP32 elementwise failure retained; tensor errors are not ATE/RPE',
                  'Only compiler flag differs; actual compiled operators/runtime results must be inspected']}
    before_affinity = os.sched_getaffinity(0)
    before_env = {k: os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}
    contexts, api, parent = {}, None, None
    try:
        parent = C.source(args.parent, PARENT_SHA, pins)
        compiler = C.source(args.compiler_source, COMPILER_SHA, pins)
        C.pin(args.compile_dir / 'compile.json', args.compile_sha256, pins)
        compiled = json.loads((args.compile_dir / 'compile.json').read_text())
        required = {'schema': 'v50_original_pair_flash_config_board_compile',
                    'status': 'both_compiled_not_runtime_validated', 'script_sha256': COMPILER_SHA,
                    'pair_graph_sha256': C.PAIR_GRAPH_SHA, 'pair_export_sha256': C.PAIR_EXPORT_SHA,
                    'actual_calibration_performed': False, 'runtime_API_called': False,
                    'candidate_accepted': False, 'source_and_inputs_unchanged': True,
                    'cwd_restored': True, 'environment_restored': True, 'cleanup_errors': []}
        if any(not C.exact(compiled.get(k),v) for k,v in required.items()):
            raise ValueError('Actual completed original graph flash compilation evidence required')
        models = {}
        for role, flag in zip(ROLES, [False, True]):
            row = compiled['variants'][role]
            if (row['status'] != 'compiled_not_runtime_validated' or row['released'] is not True
                    or not C.exact(row['configuration'], dict(compiler.CONFIG, enable_flash_attention=flag))
                    or row['build_options'] != {'do_quantization':False,'dataset':None,'auto_hybrid':False}
                    or [s['name'] for s in row['operations']] != ['config','load_onnx','build','export_rknn']
                    or any(s['status'] != 'completed' or type(s['code']) is not int or s['code'] != 0 for s in row['operations'])):
                raise ValueError('Same-toolchain flag-only nonquantizing comparison required')
            path = (args.compile_dir / row['output']['file']).resolve()
            if not path.is_relative_to(args.compile_dir.resolve()): raise ValueError('Artifact path escaped')
            models[role] = C.pin(path, row['output']['sha256'], pins)
            if path.stat().st_size != row['output']['bytes']: raise ValueError('Artifact bytes mismatch')
        CC = C.source(args.capture_source, CAPTURE_SOURCE_SHA, pins)
        captured, samples, bound = CC.load_capture(args.capture_dir, CAPTURE_SHA)
        pins.update(bound)
        evaluations = [s for s in samples if s['split'] == 'evaluation']
        if [s['sample_id'] for s in evaluations] != ['evaluation_00','evaluation_01']:
            raise ValueError('Frozen evaluation sample set required')
        report['original_checkpoint'] = captured['checkpoint']
        report['compiled_artifacts'] = {k: compiled['variants'][k]['output'] for k in ROLES}
        C.pin(args.sdk_header, C.SDK_HEADER_SHA, pins)
        library = C.pin('/usr/lib/librknnrt.so', C.SDK_LIBRARY_SHA, pins)
        if not {6,7} <= before_affinity: raise ValueError('CPU6/7 unavailable')
        os.sched_setaffinity(0,{6,7}); os.environ.update(parent.ENVIRONMENT)
        report['system_before'] = parent.snapshot(R)
        api = R.bind_api(ct.CDLL(str(library)))
        for role in ROLES:
            parent.prepare_context(R,C,api,role,models[role],contexts,report['context_records'],args.collect_perf_detail)
            if any(report['context_records'][role][d+'_attr']['type'] not in (0,1) for d in ['input','output']):
                raise ValueError('Nonquant pair must retain floating logical IO')
        import numpy as np
        for sample_index, sample in enumerate(evaluations):
            value = CC.load_array(args.capture_dir,sample['artifacts']['input.npy'],C.INPUT_SHAPE,'float32')
            reference = CC.load_array(args.capture_dir,sample['artifacts']['output.npy'],C.OUTPUT_SHAPE,'float32')
            C.validate_inputs(value,reference);value.setflags(write=False);raw=C.raw_sha(value)
            first={}
            for iteration in range(4):
                order=ROLES if (sample_index+iteration)%2==0 else ROLES[::-1]
                for role in order:
                    if C.raw_sha(value)!=raw: raise ValueError('Input changed before SDK')
                    system=parent.snapshot(R)
                    actual, sdk=parent.execute_sdk_call(R,api,contexts[role],value,3,C.OUTPUT_SHAPE,30000,
                                                       report['cleanup_errors'],report['sdk_calls'],role)
                    if C.raw_sha(value)!=raw: raise ValueError('SDK mutated input')
                    sdk.update(sample_id=sample['sample_id'],round_index=iteration,
                               sample_call_index=iteration,context_call_index=sample_index*4+iteration,
                               first_in_context=sample_index==0 and iteration==0,
                               input_raw_sha256_before=raw,input_raw_sha256_after=C.raw_sha(value))
                    first.setdefault(role,C.raw_sha(actual))
                    row={'sample_id':sample['sample_id'],'role':role,'round_index':iteration,
                         'sdk_call_index':sdk['sdk_call_index'],'input_raw_sha256':raw,
                         'reference_raw_sha256':C.raw_sha(reference), 'io_total_seconds':sdk['io_total_seconds'],
                         'original_FP32_comparison':C.compare(actual,reference),
                         'byte_equal_to_first_for_sample':C.raw_sha(actual)==first[role],
                         'system_before':system,'system_after':parent.snapshot(R),
                         'output':parent.save_array(output,f"{sample['sample_id']}-{role}-call{iteration}.npy",actual,C)}
                    if args.collect_perf_detail:
                        detail=R.perf_detail(api,contexts[role]);text=detail.pop('raw_text')
                        path=output/f"{sample['sample_id']}-{role}-call{iteration}-operators.txt"
                        path.write_text(text);detail.update(file=path.name,saved_text_sha256=C.sha(path))
                        row['separate_profile_detail']=detail
                    report['calls'].append(row)
        report['summaries']={}
        for role in ROLES:
            rows=[x for x in report['calls'] if x['role']==role and x['round_index']>0]
            if len(rows)!=6: raise ValueError('Exactly six measured repeats per implementation required')
            report['summaries'][role]={'mean_denominator':6,'subsequent_IO_mean_seconds':sum(x['io_total_seconds'] for x in rows)/6,
                                      'strict_original_FP32_gate_passed':all(x['original_FP32_comparison']['strict_FP32_gate_passed'] for x in report['calls'] if x['role']==role)}
        report['status']='completed_diagnostic_not_adopted'
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if parent is not None:
            parent.destroy_contexts(R,api,contexts,report['context_records'],report['cleanup_errors'])
            try:
                report['system_after']=parent.snapshot(R)
            except BaseException as exc:
                report['cleanup_errors'].append(f'Final platform sample: {type(exc).__name__}: {exc}')
        try:
            os.sched_setaffinity(0,before_affinity)
        except BaseException as exc:
            report['cleanup_errors'].append(f'Affinity restoration: {type(exc).__name__}: {exc}')
        for k,v in before_env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        report['affinity_restored']=os.sched_getaffinity(0)==before_affinity
        report['environment_restored']={k:os.environ.get(k) for k in before_env}==before_env
        try:
            after,unchanged=C.recheck(pins)
        except BaseException as exc:
            after,unchanged={},False
            report['cleanup_errors'].append(f'Final source hashes: {type(exc).__name__}: {exc}')
        report.update(source_and_input_hashes=pins,source_and_input_hashes_after=after,source_and_inputs_unchanged=unchanged)
        report['successful_SDK_calls']=sum(x['status']=='completed' for x in report['sdk_calls'])
        if (not unchanged or not report['affinity_restored'] or not report['environment_restored']
                or report['cleanup_errors'] or (report['status'].startswith('completed') and
                (report['successful_SDK_calls']!=16 or len(report['calls'])!=16
                 or any(x.get('context_destroyed') is not True for x in report['context_records'].values())))):
            report['status']='failed'
        (output/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'error':report.get('error'),'run_sha256':C.sha(output/'run.json')}),flush=True)
    return 0 if report['status'].startswith('completed') else 2


if __name__=='__main__':
    raise SystemExit(main())
