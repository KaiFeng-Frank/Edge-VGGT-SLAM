#!/usr/bin/env python3
"""Read completed DA3 ABBA SDK artifacts using NumPy; never run a model/SDK.

Integrity, fixed FP32 quality and A/B numerical agreement are separate results.
ABBA wall times and separate sysfs samples are observations, not a controlled
thermal/statistical benchmark. The pinned v24 auditor supplies unchanged
full-array comparison and SDK profile parsing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
V24_SHA = '5cec66e0aa99fb4997f02e668411ac37590cac47c05bb02b36e32ed75a7cba4c'
RUNNER_SHA = 'de64bcd069b78f266bdfc48c8cbf75305937a1e23ffa20eb669fe323ec8dddc8'
COMPILERS = {
    'A': '2ad1069c8d5563051120a2080bc3f79499df81d94911edb316d83b61b994f0d6',
    'B': '971184fd75f77aad0261163176e8cbc5cd1d4525d561d69c3e2d322172c075ac',
}
OLDER_COMPILERS = ('e9c66360e2de65f173682650ee8536e321acbc0698f895172d69a15cddb44a91',
                   'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5')
NORMAL_LABELS = ('A1', 'B1', 'B2', 'A2')
NORMAL_SCHEDULE = (0, 0, 0, 0, 1, 2)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_v24(path):
    if sha(path) != V24_SHA:
        raise ValueError('Frozen v24 auditor SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v25_frozen_v24_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_v24_auditor():
    for name in ('audit-v24-da3-rknn.py', 'audit-da3-npu.py'):
        path = Path(__file__).with_name(name)
        if path.exists():
            return path
    return Path('/dev/shm/vggt-v24-da3-source/audit-da3-npu.py')


def compiler_config(variant):
    if variant not in COMPILERS:
        raise ValueError('Unknown A/B variant')
    rules = ['gather_to_conv', 'fuse_channel_shuffle_split_concat_to_gather']
    if variant == 'B':
        rules.append('swap_slice_transpose')
    return dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
                do_quantization=False, disable_rules=rules)


def validate_compiler(compiled, variant, onnx_sha):
    if (compiled['script_sha256'] != COMPILERS[variant]
            or compiled['configuration'] != compiler_config(variant)
            or compiled['status'] != 'compiled_not_runtime_validated'
            or compiled['released'] is not True or compiled.get('release_error')
            or compiled['input']['sha256'] != onnx_sha
            or compiled['versions']['rknn-toolkit2'] != '2.3.2'
            or [r['name'] for r in compiled['operations']] != ['config', 'load_onnx', 'build', 'export_rknn']
            or any(r['code'] not in (0, None) for r in compiled['operations'])
            or compiled.get('frozen_v24_r1_reference_sha256') != OLDER_COMPILERS[0]
            or compiled.get('frozen_v12_reference_sha256') != OLDER_COMPILERS[1]
            or variant == 'B' and compiled.get('frozen_v24_r2_reference_sha256') != COMPILERS['A']):
        raise ValueError('Wrong completed compiler configuration/provenance: '+variant)


def warm_timing(run):
    rows = run['call_records']
    if tuple(r['pair_index'] for r in rows) != NORMAL_SCHEDULE:
        raise ValueError('Incomplete or reordered normal call schedule')
    warm = [r['forward_seconds'] for r in rows[1:4]]
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in warm):
        raise ValueError('Nonfinite/negative warm wall time')
    mean = sum(warm)/3
    if not math.isclose(mean, run['first_pair_warm_mean_seconds'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Recorded warm mean differs from actual three calls')
    return dict(cold_seconds=rows[0]['forward_seconds'], warm_seconds=warm,
                warm_mean_seconds=mean, other_pair_single_seconds=[r['forward_seconds'] for r in rows[4:]])


def merged_timing(groups):
    result = {}
    for variant, labels in (('A', ('A1', 'A2')), ('B', ('B1', 'B2'))):
        samples = [t for label in labels for t in groups[label]['warm_seconds']]
        result[variant] = dict(normal_group_labels=list(labels), warm_sample_count=len(samples),
                               six_warm_seconds=samples, pooled_warm_mean_seconds=sum(samples)/len(samples))
    a, b = result['A']['pooled_warm_mean_seconds'], result['B']['pooled_warm_mean_seconds']
    result['observed_A_over_B_ratio'] = a/b if b else None
    result['observed_relative_reduction'] = 1-b/a if a else None
    result['statistical_significance_or_controlled_speedup_claim'] = False
    return result


def platform_observation(path, variant, reader):
    if not path.exists():
        return dict(status='unknown', reason='independent platform_samples.json not supplied',
                    temperatures_or_frequencies_controlled=False)
    raw = reader(path)
    if raw.get('variant') != variant:
        raise ValueError('Platform sidecar belongs to different compiler variant')
    before, after = raw['before'], raw['after']
    if any(type(v.get('monotonic')) not in (int, float) or not math.isfinite(v['monotonic'])
           or not isinstance(v.get('sysfs'), dict) for v in (before, after)):
        raise ValueError('Malformed independent platform sample')
    if after['monotonic'] < before['monotonic']:
        raise ValueError('Reversed platform observation times')
    return dict(status='observed_separate_sidecar', file=path.name, sha256=sha(path),
                scope=raw.get('scope'), before=before, after=after,
                temperatures_or_frequencies_controlled=False,
                per_call_temperatures_or_during_run_frequency_known=False)


def sdk_profile_focus(detail):
    rows = []
    for line in detail['raw_text'].splitlines():
        m = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.*)$', line)
        if not m:
            continue
        time = re.search(r'\b\d+/\d+/\d+\s+([0-9.eE+-]+)\s', m[5])
        if m[4] == 'CPU' or '/cam_dec/' in line and m[2] == 'Transpose':
            rows.append(dict(id=int(m[1]), type=m[2], target=m[4], time_us=float(time[1]),
                             name=line.split()[-1], raw_row=line))
    return dict(all_CPU_rows=rows, camera_transpose_rows=[r for r in rows if '/cam_dec/' in r['name']
                                                       and r['type'] == 'Transpose'])


def audit(args):
    report = dict(kind='v25_DA3_completed_ABBA_saved_array_NumPy_audit', status='failed',
                  auditor_sha256=sha(__file__), frozen_v24_auditor_sha256=V24_SHA,
                  model_name='da3-small', different_model_from_VGGT=True,
                  production_changed=False, model_OR_ORT_executed_by_auditor=False,
                  board_jobs_executed_by_auditor=False, quality_validated=False,
                  real_time_validated=False, artifacts=[], numerical={}, groups={}, platform_samples={})
    failures, count = [], 0
    def check(ok, label):
        nonlocal count
        count += 1
        if not bool(ok):
            failures.append(label)
    def local_artifact(root, info, role):
        if not isinstance(info['file'], str) or Path(info['file']).name != info['file']:
            raise ValueError('Artifact must be a local filename')
        path = root/info['file']
        check(sha(path) == info['sha256'] and path.stat().st_size == info['bytes'], role+':file bytes/SHA '+path.name)
        report['artifacts'].append(dict(role=role, file=path.name, bytes=path.stat().st_size, sha256=sha(path)))
        return path
    try:
        v24 = load_v24(args.v24_auditor)
        def array(value, info, shape, label, dtype=np.float32):
            check(value.dtype == dtype and value.shape == tuple(shape) and np.isfinite(value).all(), label+':shape/dtype/finite')
            meta = v24.raw_metadata(value)
            check(all(info.get(k) == meta[k] for k in ('shape', 'dtype', 'bytes')), label+':metadata')
            hashes = [info[k] for k in ('sha256', 'tensor_sha256') if k in info]
            check(bool(hashes) and all(h == meta['tensor_sha256'] for h in hashes), label+':raw SHA')
        def metrics_equal(actual, expected, label):
            for k, value in actual.items():
                other = expected.get(k)
                ok = math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-14) if type(value) is float and type(other) in (int, float) else value == other
                check(ok, label+':'+k)
        common_paths = dict(CPU=args.cpu_dir/'capture.json', export=args.export_dir/'export.json',
                            ORT=args.ort_dir/'ort.json', compile_A=args.compile_a/'compile.json',
                            compile_B=args.compile_b/'compile.json')
        parents = {k:v24.read_json(p) for k,p in common_paths.items()}
        report['raw_receipts'] = {k:dict(file=p.name, sha256=sha(p)) for k,p in common_paths.items()}
        cpu, exp, ort = parents['CPU'], parents['export'], parents['ORT']
        check(cpu['status'] == 'completed_fixed_input_diagnostic' and cpu['model_name'] == 'da3-small', 'completed original CPU refs')
        check(exp['status'] == 'exported_not_runtime_validated' and exp['original_weights_unchanged'], 'frozen exported graph')
        check(ort['status'] == 'completed' and ort['exact_graph_reference_gate_passed'], 'preexisting exact graph CPU ORT gate')
        check(sha(args.export_dir/exp['onnx']['file']) == exp['onnx']['sha256'], 'original ONNX actual SHA')
        check(sha(args.runtime_source) == RUNNER_SHA, 'frozen current runtime source SHA')
        for variant, root in (('A', args.compile_a), ('B', args.compile_b)):
            compiled = parents['compile_'+variant]
            validate_compiler(compiled, variant, exp['onnx']['sha256'])
            model = root/Path(compiled['output']['path']).name
            check(sha(model) == compiled['output']['sha256'] and model.stat().st_size == compiled['output']['size'], variant+':actual RKNN bytes/SHA')
        refs, pixels = {}, {}
        for i, row in enumerate(cpu['records']):
            check(row['pair_index'] == i and i < 3, 'CPU source pair order')
            with np.load(local_artifact(args.cpu_dir, row['artifact'], 'CPU'), allow_pickle=False) as data:
                check(set(data.files) == set(v24.KEYS)|{'frame_ids'}, 'CPU exact canonical keys')
                refs[i] = {k:data[k].copy() for k in v24.KEYS}
                array(data['frame_ids'], row['arrays']['frame_ids'], (2,), 'CPU frame IDs', np.float64)
                check(np.array_equal(data['frame_ids'], row['frame_ids']), 'CPU source frame order')
                for key, value in refs[i].items():
                    array(value, row['arrays'][key], v24.SHAPES[key], 'CPU'+str(i)+':'+key)
            info = row['normalized_input_tensor']
            path = args.cpu_dir/info['file']
            pixels[i] = np.load(path, allow_pickle=False)
            check(sha(path) == info['file_sha256'], 'CPU normalized actual file SHA')
            array(pixels[i], info, (1,2,3,252,336), 'CPU normalized input')
        check(len(refs) == 3, 'three CPU pairs')
        for i, row in enumerate(exp['pairs']):
            info = exp['artifacts'][row['input_artifact']]
            value = np.load(local_artifact(args.export_dir, info, 'export'), allow_pickle=False)
            array(value, info['array'], (2,3,252,336), 'export normalized input')
            check(np.array_equal(value, pixels[i][0]), 'export normalized pixels unchanged')
        representatives, model_representatives, all_metrics = {}, {}, []
        timings, identity = {}, None
        profile_coverage = []
        run_items = [(label, getattr(args, 'normal_'+label.lower()), label[0], False) for label in NORMAL_LABELS]
        run_items.append(('profile_B1', args.profile_b1, 'B', True))
        for label, root, variant, is_profile in run_items:
            run_path = root/'run.json'; run = v24.read_json(run_path)
            compiled = parents['compile_'+variant]; schedule = (0,1,2) if is_profile else NORMAL_SCHEDULE
            report['raw_receipts'][label] = dict(file=run_path.name, sha256=sha(run_path))
            check(run['status'] == 'completed_fixed_input_diagnostic' and 'error' not in run, label+':completed')
            check(run['script_sha256'] == RUNNER_SHA, label+':current entrypoint pin')
            for key, parent in (('CPU_reference_capture_sha256','CPU'), ('export_sha256','export'),
                                ('ort_sha256','ORT'), ('compile_sha256','compile_'+variant)):
                check(run[key] == sha(common_paths[parent]), label+':'+key)
            check(run['compiled_model_sha256'] == compiled['output']['sha256'], label+':model identity')
            check(run['collect_perf_detail'] == is_profile and run['init_flags'] == (8 if is_profile else 0), label+':profile timing isolation')
            check(run['sdk_run_api_calls'] == run['successful_synchronous_call_count'] == len(run['call_records']) == len(schedule), label+':call counts')
            check(tuple(r['pair_index'] for r in run['call_records']) == schedule, label+':prescribed order')
            check(run['context_destroyed'] is True and not run['cleanup_errors'] and run['affinity_restored'] is True, label+':cleanup')
            check(run['sources_and_inputs_unchanged'] is True and run['source_and_input_hashes_after'] == run['source_and_input_hashes'], label+':declared unchanged sourceafter')
            check(run['cpu_affinity'] == [4,5,6,7] and run['torch_threads'] == 4 and run['core_mask'] == 7, label+':same CPU/core setup')
            current_identity = {k:run[k] for k in ('checkpoint_sha256','config_sha256','source_manifest_sha256','input_manifest_sha256','CPU_reference_capture_sha256','export_sha256','ort_sha256','sdk_version','driver_version','abi')}
            if identity is None:
                identity = current_identity
            check(current_identity == identity, label+':common model/input/source/SDK identity')
            for i in range(3):
                info = cpu['records'][i]['normalized_input_tensor']; path = root/info['file']
                check(sha(path) == info['file_sha256'], label+':normalized file SHA')
                value = np.load(path, allow_pickle=False)
                array(value, info, (1,2,3,252,336), label+':normalized input')
                check(np.array_equal(value, pixels[i]), label+':identical normalized pixels')
            log_path = root.with_suffix('.log')
            cleaned = re.sub(r'\x1b\[[0-9;]*[mK]', '', log_path.read_text())
            errors = [line for line in cleaned.splitlines() if re.search(r'\bE\s+RKNN\b|\[E:onnxruntime:', line)]
            check(not errors, label+':no illegal SDK/ORT error log')
            events = [json.loads(line) for line in cleaned.splitlines() if line.startswith('{')]
            check(len(events) == len(schedule)+1 and events[-1].get('receipt_sha256') == sha(run_path), label+':log completion/hash')
            report.setdefault('logs', {})[label] = dict(file=log_path.name, sha256=sha(log_path), errors=errors)
            calls, local_metrics = [], []
            for index, row in enumerate(run['call_records']):
                pair = row['pair_index']; key = (variant,pair); tag = label+':'+str(index)
                check(row['call_index'] == index and row['frame_ids'] == cpu['records'][pair]['frame_ids'], tag+':exact source IDs')
                check(row['input_tensor'] == cpu['records'][pair]['input_tensor'] and row['normalized_input_tensor'] == cpu['records'][pair]['normalized_input_tensor'], tag+':input receipt binding')
                check(row['synchronous'] and row['output_buffers_released'] and row['output_api_want_float'] == 1, tag+':synchronous owned output/release')
                native = run['input_attr']['format']
                expected_shape = [2,252,336,3] if native == 1 else [2,3,252,336]
                check(native in (0,1) and run['input_attr']['shape'] == expected_shape and row['input_api_format'] == native and row['input_buffer_shape'] == expected_shape and row['input_api_dtype'] == 'float32' and row['input_pass_through'] == 0, tag+':explicit correctly packed FP32 input')
                check(row['output_bytes'] == [677376,677376,72], tag+':SDK copied sizes')
                phase_keys = ('input_packing_seconds','input_set_seconds','run_api_seconds','outputs_get_seconds','output_copy_seconds','outputs_release_seconds')
                check(all(math.isfinite(row[k]) and row[k] >= 0 for k in phase_keys), tag+':finite IO phases')
                check(sum(row[k] for k in phase_keys) <= row['sdk_io_seconds']+1e-5 and math.isclose(row['forward_seconds'], row['sdk_io_seconds']+row['official_codec_seconds'], abs_tol=1e-12), tag+':whole IO/codec timing')
                path = local_artifact(root, row['artifact'], label)
                with np.load(path, allow_pickle=False) as data:
                    check(set(data.files) == set(v24.KEYS)|{'frame_ids'}, tag+':exact canonical keys')
                    array(data['frame_ids'], row['arrays']['frame_ids'], (2,), tag+':stored IDs', np.float64)
                    check(np.array_equal(data['frame_ids'], row['frame_ids']), tag+':stored source order')
                    metrics = {}
                    for k in v24.KEYS:
                        value = data[k]
                        array(value, row['arrays'][k], v24.SHAPES[k], tag+':'+k)
                        metrics[k] = v24.compare(value, refs[pair][k])
                        metrics_equal(metrics[k], row['versus_CPU_reference'][k], tag+':'+k+':CPU comparison')
                        check(not metrics[k]['actual_all_zero'], tag+':'+k+':nonzero')
                        if k in ('depth','depth_conf','pose_enc'):
                            raw = value[0,...,0] if k == 'depth' else value[0]
                            array(raw, row['raw_outputs'][k], (2,252,336) if k != 'pose_enc' else (2,9), tag+':raw '+k)
                        if key in model_representatives:
                            check(value.tobytes(order='C') == model_representatives[key][k].tobytes(order='C'), tag+':'+k+':same-model repeated/profile raw bytes')
                    if key not in model_representatives:
                        model_representatives[key] = {k:data[k].copy() for k in v24.KEYS}
                    representatives[(label,pair)] = {k:data[k].copy() for k in v24.KEYS}
                local_metrics.append(metrics); all_metrics.append(metrics)
                calls.append(dict(call_index=index,pair_index=pair, forward_seconds=row['forward_seconds'],
                                  sdk_io_seconds=row['sdk_io_seconds'],codec_seconds=row['official_codec_seconds'],
                                  arrays_SHA256={k:row['arrays'][k]['tensor_sha256'] for k in v24.KEYS}))
                if is_profile:
                    check(row['perf_detail']['query_command'] == 3, tag+':actual perf query')
                    profile_coverage.append(dict(pair_index=pair,**v24.parse_profile(row['perf_detail']),**sdk_profile_focus(row['perf_detail'])))
            gate = all(m['elementwise_gate_passed'] for value in local_metrics for m in value.values())
            check(run['exact_FP32_reference_gate_passed'] == gate, label+':strict gate honesty')
            report['groups'][label] = dict(variant=variant, normal_benchmark=not is_profile, sdk_calls=len(calls),
                calls=calls, exact_FP32_reference_gate_passed=gate, context_destroyed=run['context_destroyed'],
                cleanup_errors=run['cleanup_errors'], affinity_restored=run['affinity_restored'],
                model_memory_bytes=run['model_memory_bytes'], memory_final=run['memory_final'],
                sourceafter_claim_preserved=bool(run['sources_and_inputs_unchanged'])
                and run['source_and_input_hashes_after'] == run['source_and_input_hashes'])
            if not is_profile:
                timings[label] = warm_timing(run)
            report['platform_samples'][label] = platform_observation(root/'platform_samples.json',variant,v24.read_json)
        report['numerical']['CPU_by_model_and_pair'] = {variant:{str(i):{k:v24.compare(model_representatives[(variant,i)][k], refs[i][k]) for k in v24.KEYS} for i in range(3)} for variant in ('A','B')}
        report['numerical']['B_versus_A_by_pair'] = {str(i):{k:v24.compare(model_representatives[('B',i)][k], model_representatives[('A',i)][k]) for k in v24.KEYS} for i in range(3)}
        report['numerical']['A_B_all_pairs_bitwise_equal'] = all(m['C_order_bytes_equal'] for p in report['numerical']['B_versus_A_by_pair'].values() for m in p.values())
        report['numerical']['all_CPU_comparisons_fixed_gate_passed'] = all(m['elementwise_gate_passed'] for row in all_metrics for m in row.values())
        report['full_pixel_arrays_recomputed'] = len(all_metrics)*len(v24.KEYS)
        report['timings'] = dict(normal_groups=timings, pooled_observations=merged_timing(timings),
            includes='packing + synchronous SDK set/run/get/copy/release + official CPU codec',
            excludes='load/init; attribute/performance queries; finite/error comparisons; hashing; saving',
            profile_excluded_from_normal_timings=True)
        report['operator_coverage'] = dict(B1_profile=profile_coverage, all_internal_on_NPU=all(p['internal_CPU_layers']==0 and p['targets'].get('GPU',0)==0 for p in profile_coverage))
        if args.old_profile_dir.exists():
            previous_path = args.old_profile_dir/'run.json'; previous = v24.read_json(previous_path)
            check(previous['compiled_model_sha256'] == parents['compile_A']['output']['sha256'] and previous['collect_perf_detail'] is True, 'old A profile model/isolation')
            report['operator_coverage']['old_A_profile'] = dict(receipt_SHA256=sha(previous_path),
                different_process_runtime_and_sampling_state=True,
                calls=[dict(pair_index=r['pair_index'],**v24.parse_profile(r['perf_detail']),**sdk_profile_focus(r['perf_detail'])) for r in previous['call_records']])
        else:
            report['operator_coverage']['old_A_profile'] = dict(status='unknown',reason='old profile not supplied')
        report['identity'] = dict(runtime_SHA256=RUNNER_SHA, compiler_SHA256=COMPILERS,
                                 ONNX_SHA256=exp['onnx']['sha256'],
                                 model_SHA256={v:parents['compile_'+v]['output']['sha256'] for v in ('A','B')},
                                 checkpoint_SHA256=identity['checkpoint_sha256'])
        report['limits'] = ['Passed means completed-artifact/receipt/log consistency; strict FP32 and task quality remain separate.',
            'Only completed local artifacts were read. Absolute board source-after hashes are receipt claims, not reread board files.',
            'A/B values are compared without relaxing fixed rtol=1e-4/atol=1e-5. Same-model repeats/profile must preserve bytes.',
            'ABBA uses four separate processes and only three pair0 warm calls per process; no statistical significance, sustained throughput or online latency claim.',
            'Platform sidecars are independent before/after read-only sysfs samples, not per-call temperatures or frequency/governor control.',
            'Layer times and counts are separate profile diagnostics. The previous A profile is not a simultaneous controlled timing baseline.',
            'S=2 produces two poses from one overlapping pair; this is not two independent new observations or 20 Hz validation.']
    except Exception as error:
        failures.append(type(error).__name__+': '+str(error))
    report['checks'] = dict(count=count,failures=failures)
    report['status'] = 'passed' if not failures else 'failed'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=Path('/dev/shm'))
    parser.add_argument('--v24-auditor',type=Path,default=default_v24_auditor())
    parser.add_argument('--runtime-source',type=Path,default=Path(__file__).with_name('run-v25-da3-rknn-board.py'))
    for flag in ('normal-a1','normal-b1','normal-b2','normal-a2','profile-b1','cpu-dir','export-dir','ort-dir','compile-a','compile-b','old-profile-dir'):
        parser.add_argument('--'+flag,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    defaults=dict(normal_a1='vggt-v25-da3-npu-normal-A1',normal_b1='vggt-v25-da3-npu-normal-B1',
                  normal_b2='vggt-v25-da3-npu-normal-B2',normal_a2='vggt-v25-da3-npu-normal-A2',
                  profile_b1='vggt-v25-da3-npu-profile-B1',cpu_dir='vggt-v24-da3-cpu-r2',
                  export_dir='vggt-v24-da3-export-r4',ort_dir='vggt-v24-da3-ort-r4b',
                  compile_a='vggt-v24-da3-compile-r4c',compile_b='vggt-v25-da3-compile-r1',
                  old_profile_dir='vggt-v24-da3-npu-profile-r1')
    for name,folder in defaults.items():
        if getattr(args,name) is None:
            setattr(args,name,args.run_root/folder)
    if args.output.exists():
        parser.error('New output required; refusing overwrite')
    report=audit(args)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report['checks'],output=str(args.output),sha256=sha(args.output))))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
