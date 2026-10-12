#!/usr/bin/env python3
"""Board-only original frame0 QKV layout test; byte-identical FP32 gate before FP16 build."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
HELPER_SHA = '4094e393d0504954a67993c24e82c5534862393930f60db3ec64c2c3e4098b76'
REWRITE_SHA = '2bc71dcb1b7c277681cb6121250d7255cb18c499ab7d2bd5e6c7a7a35c8608e3'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
BINDINGS = [dict(index=0, name='frame_output', shape=[2, 53, 1024], dtype='float32'),
            dict(index=1, name='global_output', shape=[1, 106, 1024], dtype='float32')]


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''): h.update(data)
    return h.hexdigest()


def metadata(path):
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def check(c, baseline, candidate, cases, output, report):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    row = dict(schema='v75_original_VGGT_pair00_frame_QKV_head_major_folded_ONNX_check', status='running',
        original_parent_manifest_sha256=PARENT_SHA, original_pair_onnx_sha256=GRAPH_SHA,
        candidate_onnx_sha256=sha(candidate), script_sha256=report['script_sha256'],
        rewrite_helper_sha256=REWRITE_SHA, graph_proof=report['graph_proof'], cases=cases,
        ORT_version=ort.__version__, optimization='ORT_DISABLE_ALL', intra_op_num_threads=1,
        inter_op_num_threads=1, providers=['CPUExecutionProvider'], candidate_accepted=False,
        cleanup_errors=[], timing_is_not_NPU_performance=True, original_dot_K_reduction_order_unchanged=True)
    sessions = []
    c.write(output / 'check.json', row)
    try:
        for path in (baseline, candidate):
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            sessions.append(session); session.disable_fallback()
            require(session.get_providers() == ['CPUExecutionProvider']
                and [(v.name, v.shape, v.type) for v in session.get_inputs()] == [('input', [2,53,1024], 'tensor(float)')]
                and [(v.name, v.shape, v.type) for v in session.get_outputs()] == [(b['name'],b['shape'],'tensor(float)') for b in BINDINGS], 'ORT ABI/provider differs')
        for case in cases:
            root = Path(case['directory']); x = c.tensor(np, root, case['input'], [2,53,1024])
            case['original_outputs'] = case['original_outputs'][:2] if case['original_outputs'] else None
            case.update(saved_input=c.save_tensor(np, output, case['tag']+'-input.npy', x, dict(name='input',shape=[2,53,1024])),
                        baseline_outputs=[], candidate_outputs=[], checks=[])
            ys = []
            for label, session in zip(('baseline', 'candidate'), sessions):
                started = time.perf_counter(); ys.append(session.run([b['name'] for b in BINDINGS], {'input':x}))
                case[label+'_ORT_CPU_seconds'] = time.perf_counter()-started
            for i, binding in enumerate(BINDINGS):
                a, b = ys[0][i], ys[1][i]
                for label, value in (('baseline',a), ('candidate',b)):
                    case[label+'_outputs'].append(c.save_tensor(np, output, f"{case['tag']}-{label}-{i}.npy", value, binding))
                item = dict(name=binding['name'], byte_equal_to_baseline=a.tobytes()==b.tobytes(),
                            versus_baseline_FP32_ONNX=c.compare(np,b,a), versus_original_FP32=None)
                if case['original_outputs']:
                    reference = c.tensor(np, root, case['original_outputs'][i], binding['shape'])
                    item['versus_original_FP32'] = {label:c.compare(np,value,reference) for label,value in (('baseline',a),('candidate',b))}
                case['checks'].append(item)
            c.write(output / 'check.json', row)
        row['all_candidate_vs_baseline_FP32_ONNX_byte_equal'] = all(v['byte_equal_to_baseline'] for case in cases for v in case['checks'])
        row['all_candidate_vs_baseline_FP32_ONNX_strict_passed'] = all(v['versus_baseline_FP32_ONNX']['strict_pass'] for case in cases for v in case['checks'])
        row['original_Torch_gate_passed_where_reference_available'] = all(v['versus_original_FP32'][label]['strict_pass']
            for case in cases for v in case['checks'] if v['versus_original_FP32'] for label in ('baseline','candidate'))
        row['original_Torch_reference_case_count'] = sum(case['original_outputs'] is not None for case in cases)
        row['original_Torch_references_cover_all_cases'] = all(case['original_outputs'] is not None for case in cases)
        require(row['all_candidate_vs_baseline_FP32_ONNX_byte_equal'] and row['all_candidate_vs_baseline_FP32_ONNX_strict_passed'],
                'QKV layout differs from original FP32 ONNX bytes; compile blocked')
        row['status'] = 'passed_original_FP32_ONNX_byte_equivalence_not_quality_accepted'
    except BaseException as exc:
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        sessions.clear(); session = None; row['sessions_dropped'] = True; c.write(output/'check.json', row)
    report.update(cases=cases, check_receipt_sha256=sha(output/'check.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline-manifest',type=Path,default=BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--capture-dir',type=Path,default=BASE/'experiments/vggt-v53-capture-r1')
    p.add_argument('--calibration-dir',type=Path,default=BASE/'experiments/vggt-v56-calibration-r1')
    p.add_argument('--reference-helper',type=Path,default=BASE/'src/deploy/rk3588/validation/original_pair_group/check_compile_pair_group.py')
    p.add_argument('--rewrite-helper',type=Path,default=Path(__file__).with_name('rewrite_frame_qkv_fold_heads.py'))
    p.add_argument('--check-only',action='store_true'); a = p.parse_args()
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board required')
    require(sha(a.reference_helper)==HELPER_SHA and sha(a.rewrite_helper)==REWRITE_SHA,'Frozen source helper changed')
    c = module(a.reference_helper, '_original_pair_cases'); q = module(a.rewrite_helper, '_original_frame_qkv')
    output = a.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    r = dict(schema='v75_original_VGGT_pair00_frame_QKV_head_major_folded_build',status='running',script_sha256=sha(__file__),
        helper_sha256=HELPER_SHA,rewrite_helper_sha256=REWRITE_SHA,original_parent_manifest_sha256=PARENT_SHA,
        original_pair_onnx_sha256=GRAPH_SHA,output_bindings=BINDINGS,input_shape=[2,53,1024],
        candidate_accepted=False,quality_accepted=False,realtime_validated=False,original_model_only=True,
        configuration=c.CONFIG,build_options=c.BUILD,operations=[],released=None,cleanup_errors=[],
        original_dot_K_reduction_order_unchanged=True,compiled_rounding_equivalence_validated=False)
    watched = [Path(__file__).resolve(),a.reference_helper.resolve(),a.rewrite_helper.resolve(),
        a.rewrite_helper.with_name('rewrite_frame_qkv_head_major.py').resolve()]; pins = {}; sdk = None
    cwd, tmp, affinity, started = Path.cwd(),os.environ.get('TMPDIR'),os.sched_getaffinity(0),time.perf_counter()
    r['cpu_affinity_before'] = sorted(affinity); c.write(output/'result.json',r)
    try:
        manifest = c.pin(a.baseline_manifest,PARENT_SHA); parent = json.loads(manifest.read_text()); row = parent['blocks'][0]
        require(parent['schema']=='v55_original_VGGT_24_frame_global_pair_RKNN_manifest' and parent['checkpoint']==c.CHECKPOINT
            and row['role']=='pair' and row['index']==0 and row['sha256']==GRAPH_SHA and row['output_bindings']==BINDINGS,'Exact original pair00 required')
        graph = c.artifact(manifest.parent,row); watched.extend((manifest,graph))
        cases = c.real_cases(a.capture_dir.resolve(strict=True),a.calibration_dir.resolve(strict=True),parent,watched)
        require(len(cases)==2 and [case['tag'] for case in cases]==['v53','v56-calib00'],'Exactly two distinct frozen real inputs required')
        pins = {str(path):sha(path) for path in watched}; r.update(parent=row,checkpoint=parent['checkpoint'],source_and_input_hashes=pins)
        import onnx
        original = onnx.load(str(graph),load_external_data=False); rewritten,proof = q.rewrite(original,row)
        require(proof['original_pair_onnx_sha256']==GRAPH_SHA and proof['role']=='frame'
            and all(proof[k] is True for k in ('all_other_original_nodes_protobuf_unchanged','all_other_original_initializers_protobuf_unchanged',
                'original_IO_protobuf_unchanged','global0_protobuf_unchanged','original_dot_K_reduction_order_unchanged',
                'all_original_parameter_values_preserved_by_inverse_permutation')),'Original QKV layout proof differs')
        candidate = output/'pair-frame-qkv-head-major.onnx'; onnx.checker.check_model(rewritten,full_check=True); onnx.save_model(rewritten,str(candidate))
        require(proof['variant']=='head_major_with_frame_head_view_folding' and proof['restored_before_original_attention_heads_transpose'] is True, 'Folded attention restoration proof required')
        r.update(graph_proof=proof,onnx=metadata(candidate)); pins[str(candidate)] = r['onnx']['sha256']
        del original,rewritten
        check(c,graph,candidate,cases,output,r); r['status']='checked_not_quality_accepted'; c.write(output/'result.json',r)
        if not a.check_only:
            version = importlib.metadata.version('rknn-toolkit2'); require(version.split('+')[0]=='2.3.2','RKNN2.3.2 required'); r['toolkit_version']=version
            os.chdir(output); os.environ['TMPDIR']=str(output)
            from rknn.api import RKNN
            sdk = RKNN(verbose=True,verbose_file=str(output/'sdk.log')); r['released']=False
            model = output/'pair-frame-qkv-head-major.rknn'
            for name,fn in (('config',lambda:sdk.config(**c.CONFIG)),('load_onnx',lambda:sdk.load_onnx(model=str(candidate))),
                            ('build',lambda:sdk.build(**c.BUILD)),('export_rknn',lambda:sdk.export_rknn(str(model)))):
                operation=dict(name=name,status='running'); r['operations'].append(operation); c.write(output/'result.json',r); tick=time.perf_counter()
                try:
                    code=fn(); operation.update(status='completed',code=code); require(code in (None,0),f'{name} returned {code}')
                finally:
                    operation['seconds']=time.perf_counter()-tick; c.write(output/'result.json',r)
            r.update(rknn=metadata(model),status='compiled_not_quality_accepted')
    except BaseException as exc:
        r.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if r['operations'] and r['operations'][-1]['status']=='running': r['operations'][-1].update(status='failed',error=r['error'])
    finally:
        if sdk is not None:
            try:
                r['release_code']=sdk.release(); r['released']=r['release_code'] in (None,0); require(r['released'],'SDK release failed')
            except BaseException as exc: r['cleanup_errors'].append(f'release: {type(exc).__name__}: {exc}')
        try:
            os.chdir(cwd)
            if tmp is None: os.environ.pop('TMPDIR',None)
            else: os.environ['TMPDIR']=tmp
            r['cpu_affinity_after']=sorted(os.sched_getaffinity(0)); r['cpu_affinity_unchanged']=os.sched_getaffinity(0)==affinity
            r['environment_restored']=Path.cwd()==cwd and os.environ.get('TMPDIR')==tmp
            r['source_and_input_hashes_after']={path:sha(Path(path)) for path in pins}
            r['source_and_inputs_unchanged']=bool(pins) and pins==r['source_and_input_hashes_after']
            require(r['environment_restored'] and r['cpu_affinity_unchanged'] and r['source_and_inputs_unchanged'],'Source/input/environment/affinity changed')
        except BaseException as exc: r['cleanup_errors'].append(f'restore/identity: {type(exc).__name__}: {exc}')
        if r['cleanup_errors']: r['status']='failed'
        r['seconds']=time.perf_counter()-started; c.write(output/'result.json',r)
        print(json.dumps(dict(status=r['status'],result_sha256=sha(output/'result.json'))),flush=True)
    return int(r['status']=='failed')


if __name__=='__main__':
    raise SystemExit(main())
