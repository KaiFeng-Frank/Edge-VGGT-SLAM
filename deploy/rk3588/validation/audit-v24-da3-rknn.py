#!/usr/bin/env python3
"""Audit saved DA3 RKNN arrays and SDK reports using NumPy only.

This reads completed files. It does not import Torch/ONNX/SDK, execute a model,
or prove task quality. A passed audit can retain a failed strict FP32 gate and
internal CPU operators. CPU/export/ORT source audits remain separate evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
RTOL, ATOL = 1e-4, 1e-5
KEYS = ('depth', 'depth_conf', 'pose_enc', 'extrinsic', 'intrinsic')
SHAPES = dict(depth=(1, 2, 252, 336, 1), depth_conf=(1, 2, 252, 336),
              pose_enc=(1, 2, 9), extrinsic=(1, 2, 3, 4), intrinsic=(1, 2, 3, 3))
RUNNER_SHA = '455770d23192dc9218c31ee352c0fc609d08408d80e3aa1b291c94ae6d83b9c1'
COMPILER_SHA = '2ad1069c8d5563051120a2080bc3f79499df81d94911edb316d83b61b994f0d6'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def raw_metadata(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape), dtype=str(array.dtype), bytes=int(array.nbytes),
                tensor_sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (
        _ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.dtype != np.float32:
        raise ValueError('Matching FP32 arrays required')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite saved output/reference')
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    delta = a-b
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
    return dict(max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta*delta))),
                relative_l2=float(np.linalg.norm(delta)/nb) if nb else None,
                cosine=float(np.dot(a/na, b/nb)) if na and nb else None,
                array_equal=bool(np.array_equal(actual, expected)),
                C_order_bytes_equal=actual.tobytes(order='C') == expected.tobytes(order='C'),
                element_count=int(close.size), failed_elements=int(np.count_nonzero(~close)),
                elementwise_gate_passed=bool(close.all()), rtol=RTOL, atol=ATOL,
                actual_all_zero=bool(np.all(actual == 0)), finite=True)


def parse_profile(detail):
    text = detail['raw_text']
    raw = text.encode('utf-8')
    if len(raw) != detail['bytes'] or hashlib.sha256(raw).hexdigest() != detail['raw_sha256']:
        raise ValueError('Profile raw bytes/SHA mismatch')
    rows = []
    for line in text.splitlines():
        match = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.*)$', line)
        if not match:
            continue
        clock = re.search(r'\b\d+/\d+/\d+\s+([0-9.eE+-]+)\s', match.group(5))
        if clock is None:
            raise ValueError('Unparsed SDK operator time')
        rows.append(dict(id=int(match.group(1)), type=match.group(2), dtype=match.group(3),
                         target=match.group(4), time_us=float(clock.group(1)), name=line.split()[-1]))
    if not rows or [row['id'] for row in rows] != list(range(1, len(rows)+1)):
        raise ValueError('Missing/duplicate SDK layer rows')
    if any(not math.isfinite(row['time_us']) or row['time_us'] < 0 for row in rows):
        raise ValueError('Invalid SDK layer time')
    # The SDK also prints prose "Total Operator ..." and "Total Memory ..."
    # lines. Only this numeric row belongs to the CPU/GPU/NPU summary table.
    number = r'[0-9.eE+-]+'
    totals = [match.groups() for line in text.splitlines()
              if (match := re.fullmatch(r'\s*Total\s+(' + number + r')\s+(' + number +
                                         r')\s+(' + number + r')\s+(' + number + r')\s*', line))]
    if len(totals) != 1:
        raise ValueError('Missing profile Total row')
    for target, total in zip(('CPU', 'GPU', 'NPU'), map(float, totals[0][:3])):
        if not math.isclose(sum(r['time_us'] for r in rows if r['target'] == target), total, abs_tol=1.0):
            raise ValueError('Profile layer times differ from declared target totals')
    counts = dict(Counter(row['target'] for row in rows))
    io = [r for r in rows if r['type'] in ('InputOperator', 'OutputOperator')]
    cpu = [r for r in rows if r['target'] == 'CPU' and r not in io]
    return dict(raw_bytes=len(raw), raw_sha256=detail['raw_sha256'], layer_count=len(rows),
                targets=counts, CPU_IO_layers=len([r for r in io if r['target'] == 'CPU']),
                internal_CPU_layers=len(cpu),
                types_by_target={target:dict(Counter(r['type'] for r in rows if r['target'] == target)) for target in counts},
                target_time_us={target:sum(r['time_us'] for r in rows if r['target'] == target) for target in counts},
                internal_CPU_time_us=sum(r['time_us'] for r in cpu),
                top_operators=sorted(rows, key=lambda row:row['time_us'], reverse=True)[:10])


def audit(args):
    failures, checked_count = [], 0
    result = dict(kind='da3_saved_RKNN_NumPy_audit_v24', status='failed', model_name='da3-small',
                  different_model_from_VGGT=True, original_VGGT_weights_used=False,
                  reviewer_model_OR_ORT_executed=False, reviewer_board_jobs_executed=False,
                  quality_validated=False, real_time_validated=False, online_pipeline_validated=False,
                  auditor_sha256=sha(__file__), artifacts=[], numerical={}, timings={}, operator_coverage=[])
    def check(value, name):
        nonlocal checked_count
        checked_count += 1
        if not bool(value):
            failures.append(name)
    def artifact(root, item):
        name = item['file']
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Nonlocal artifact path')
        path = root/name
        digest = sha(path)
        check(digest == item['sha256'] and path.stat().st_size == item['bytes'], 'artifact:'+root.name+'/'+name)
        result['artifacts'].append(dict(role=root.name, file=name, bytes=path.stat().st_size, sha256=digest))
        return path
    def array(value, info, shape, name, dtype=np.float32):
        check(value.dtype == dtype and value.shape == tuple(shape) and np.isfinite(value).all(), name+':shape/dtype/finite')
        actual = raw_metadata(value)
        check(all(info.get(k) == actual[k] for k in ('shape','dtype','bytes')), name+':metadata')
        hashes = [info[k] for k in ('sha256','tensor_sha256') if k in info]
        check(bool(hashes) and all(h == actual['tensor_sha256'] for h in hashes), name+':rawSHA')
    def metrics_equal(first, second, name):
        for key, value in first.items():
            other = second.get(key)
            equal = math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-14) if type(value) is float and type(other) in (float,int) else value == other
            check(equal, name+':'+key)
    try:
        paths = dict(normal=args.normal_dir/'run.json', profile=args.profile_dir/'run.json',
                     CPU=args.cpu_dir/'capture.json', export=args.export_dir/'export.json',
                     compile=args.compile_dir/'compile.json', ORT=args.ort_dir/'ort.json')
        receipts = {key:read_json(path) for key,path in paths.items()}
        result['raw_receipts'] = {key:dict(file=path.name, sha256=sha(path)) for key,path in paths.items()}
        cpu, exported, compiled, ort = (receipts[k] for k in ('CPU','export','compile','ORT'))
        check(cpu['status'] == 'completed_fixed_input_diagnostic' and cpu['model_name'] == 'da3-small', 'CPU completed DA3 identity')
        check(exported['status'] == 'exported_not_runtime_validated' and exported['original_weights_unchanged'], 'export status/state')
        check(ort['status'] == 'completed' and ort['exact_graph_reference_gate_passed'], 'preexisting board ORT gate')
        check(compiled['status'] == 'compiled_not_runtime_validated' and compiled['released'], 'compiler completed/released')
        check(compiled['script_sha256'] == COMPILER_SHA and compiled['configuration'] == dict(target_platform='rk3588', float_dtype='float16', optimization_level=3, do_quantization=False,
            disable_rules=['gather_to_conv','fuse_channel_shuffle_split_concat_to_gather']), 'compiler exact rules/config')
        graph = args.export_dir/exported['onnx']['file']
        check(sha(graph) == exported['onnx']['sha256'] == compiled['input']['sha256'], 'ONNX file/compile binding')
        model = args.compile_dir/Path(compiled['output']['path']).name
        check(sha(model) == compiled['output']['sha256'] and model.stat().st_size == compiled['output']['size'], 'RKNN file bytes/SHA')
        check(sha(args.runtime_source) == RUNNER_SHA, 'reviewed runtime source SHA')
        reference_arrays, normalized = {}, {}
        for i,row in enumerate(cpu['records']):
            check(row['pair_index'] == i and i < 3, 'CPU reference pair order')
            path = artifact(args.cpu_dir, row['artifact'])
            with np.load(path, allow_pickle=False) as saved:
                reference_arrays[i] = {key:saved[key].copy() for key in KEYS}
                array(saved['frame_ids'], row['arrays']['frame_ids'], (2,), f'CPU{i}:frame_ids', np.float64)
                check(np.array_equal(saved['frame_ids'], row['frame_ids']), f'CPU{i}:frame source order')
            for key,value in reference_arrays[i].items():
                array(value, row['arrays'][key], SHAPES[key], f'CPU{i}:{key}')
            info = row['normalized_input_tensor']
            normalized[i] = np.load(args.cpu_dir/info['file'], allow_pickle=False)
            check(sha(args.cpu_dir/info['file']) == info['file_sha256'], f'CPU{i}:normalized file SHA')
            array(normalized[i], info, (1,2,3,252,336), f'CPU{i}:normalized')
        check(len(reference_arrays) == 3, 'all three CPU reference pairs')
        for i, row in enumerate(exported['pairs']):
            exp = exported['artifacts'][row['input_artifact']]
            pixels = np.load(artifact(args.export_dir, exp), allow_pickle=False)
            array(pixels, exp['array'], (2,3,252,336), f'export{i}:normalized')
            check(np.array_equal(pixels, normalized[i][0]), f'export{i}:same normalized pixels')
        representatives, all_metrics, call_rows = {}, [], []
        for role, root, expected_schedule, log_path in (
            ('normal',args.normal_dir,[0,0,0,0,1,2],args.normal_log),
            ('profile',args.profile_dir,[0,1,2],args.profile_log)):
            run = receipts[role]
            check(run['status'] == 'completed_fixed_input_diagnostic' and 'error' not in run, role+':completed')
            check(run['script_sha256'] == RUNNER_SHA, role+':runtime pin')
            for key, parent in [('CPU_reference_capture_sha256','CPU'),('export_sha256','export'),('compile_sha256','compile'),('ort_sha256','ORT')]:
                check(run[key] == sha(paths[parent]), role+':'+key)
            check(run['compiled_model_sha256'] == compiled['output']['sha256'], role+':model pin')
            check(run['collect_perf_detail'] == (role == 'profile') and run['init_flags'] == (8 if role == 'profile' else 0), role+':profile isolation')
            check(run['sdk_run_api_calls'] == run['successful_synchronous_call_count'] == len(run['call_records']) == len(expected_schedule), role+':call counts')
            check([row['pair_index'] for row in run['call_records']] == expected_schedule, role+':exact call schedule')
            check(run['context_destroyed'] is True and not run['cleanup_errors'] and run['affinity_restored'] is True, role+':cleanup')
            check(run['sources_and_inputs_unchanged'] is True and run['source_and_input_hashes_after'] == run['source_and_input_hashes'], role+':declared sourceafter')
            check(run['cpu_affinity'] == [4,5,6,7] and run['torch_threads'] == 4 and run['core_mask'] == 7, role+':CPU/core configuration')
            check(run['input_attr']['format'] == 1 and run['input_attr']['shape'] == [2,252,336,3], role+':native NHWC input')
            for i in range(3):
                info = cpu['records'][i]['normalized_input_tensor']
                path = root/info['file']
                check(sha(path) == info['file_sha256'], role+f':normalized{i} fileSHA')
                pixels = np.load(path, allow_pickle=False)
                array(pixels, info, (1,2,3,252,336), role+f':normalized{i}')
                check(np.array_equal(pixels, normalized[i]), role+f':normalized{i} equal CPU')
            raw_log = log_path.read_text()
            cleaned = re.sub(r'\x1b\[[0-9;]*[mK]', '', raw_log)
            errors = [line for line in cleaned.splitlines() if re.search(r'\bE\s+RKNN\b|\[E:onnxruntime:', line)]
            check(not errors, role+':no illegal SDK/ORT error log')
            log_events = [json.loads(line) for line in cleaned.splitlines() if line.startswith('{')]
            check(len(log_events) == len(expected_schedule)+1 and log_events[-1].get('receipt_sha256') == sha(paths[role]), role+':log events/finalSHA')
            result.setdefault('logs', {})[role] = dict(file=log_path.name, sha256=sha(log_path), errors=errors)
            for index, row in enumerate(run['call_records']):
                i = row['pair_index']; label = role+f':call{index}'
                check(row['call_index'] == index and row['frame_ids'] == cpu['records'][i]['frame_ids'], label+':source identity')
                check(row['input_tensor'] == cpu['records'][i]['input_tensor'] and row['normalized_input_tensor'] == cpu['records'][i]['normalized_input_tensor'], label+':input metadata')
                check(row['synchronous'] and row['output_buffers_released'] and row['output_api_want_float'] == 1, label+':synchronous owned-output/release')
                check(row['input_api_format'] == 1 and row['input_buffer_shape'] == [2,252,336,3] and row['input_api_dtype'] == 'float32' and row['input_pass_through'] == 0, label+':explicit FP32 NHWC transport')
                check(row['output_bytes'] == [677376,677376,72], label+':SDK FP32 output sizes')
                phase_keys = ('input_packing_seconds','input_set_seconds','run_api_seconds','outputs_get_seconds','output_copy_seconds','outputs_release_seconds')
                check(all(math.isfinite(row[k]) and row[k] >= 0 for k in phase_keys), label+':finite phase timings')
                check(sum(row[k] for k in phase_keys) <= row['sdk_io_seconds']+1e-5, label+':IO phases within total')
                check(math.isclose(row['forward_seconds'], row['sdk_io_seconds']+row['official_codec_seconds'], abs_tol=1e-12), label+':forward includes codec')
                path = artifact(root, row['artifact'])
                with np.load(path, allow_pickle=False) as saved:
                    check(set(saved.files) == set(KEYS)|{'frame_ids'}, label+':exact NPZ keys')
                    array(saved['frame_ids'], row['arrays']['frame_ids'], (2,), label+':frameIDs', np.float64)
                    check(np.array_equal(saved['frame_ids'], row['frame_ids']), label+':stored frameIDs')
                    metrics = {}
                    for key in KEYS:
                        actual = saved[key]
                        array(actual, row['arrays'][key], SHAPES[key], label+':'+key)
                        metrics[key] = compare(actual, reference_arrays[i][key])
                        metrics_equal(metrics[key], row['versus_CPU_reference'][key], label+':'+key+':comparison')
                        check(not metrics[key]['actual_all_zero'], label+':'+key+':not all zero')
                        if key in ('depth','depth_conf','pose_enc'):
                            raw = actual[0,...,0] if key == 'depth' else actual[0]
                            array(raw, row['raw_outputs'][key], (2,252,336) if key != 'pose_enc' else (2,9), label+':raw '+key)
                        if i in representatives:
                            check(np.array_equal(actual, representatives[i][key]), label+':'+key+':same-pair repeat/profile equality')
                    if i not in representatives:
                        representatives[i] = {key:saved[key].copy() for key in KEYS}
                        result['numerical'][str(i)] = dict(frame_ids=row['frame_ids'], versus_CPU_reference=metrics)
                all_metrics.append(metrics)
                call_rows.append(dict(role=role, call_index=index, pair_index=i, file=path.name,
                    sdk_io_seconds=row['sdk_io_seconds'], codec_seconds=row['official_codec_seconds'],
                    forward_seconds=row['forward_seconds'], runtime_frame_id=row['runtime_frame_id'],
                    raw_array_SHA256={key:row['arrays'][key]['tensor_sha256'] for key in KEYS}))
                if role == 'profile':
                    check(row['perf_detail']['query_command'] == 3, label+':actual perf query')
                    result['operator_coverage'].append(dict(pair_index=i, **parse_profile(row['perf_detail'])))
            observed_gate = all(m['elementwise_gate_passed'] for c in all_metrics[-len(expected_schedule):] for m in c.values())
            check(run['exact_FP32_reference_gate_passed'] == observed_gate, role+':strict gate recorded honestly')
        check(receipts['normal']['source_and_input_hashes'] == receipts['profile']['source_and_input_hashes'], 'normal/profile source pins identical')
        result['all_calls'] = call_rows
        result['full_pixel_arrays_recomputed'] = len(all_metrics)*len(KEYS)
        result['same_pair_repeated_outputs_bitwise_equal'] = not any('repeat/profile equality' in name for name in failures)
        result['exact_FP32_reference_gate_passed'] = all(m['elementwise_gate_passed'] for c in all_metrics for m in c.values())
        normal = receipts['normal']
        warm = [r['forward_seconds'] for r in normal['call_records'][1:4]]
        warm_mean = sum(warm)/3
        check(math.isclose(warm_mean, normal['first_pair_warm_mean_seconds'], abs_tol=1e-12), 'normal mean independently recomputed')
        result['timings'] = dict(normal_cold_forward_seconds=normal['call_records'][0]['forward_seconds'],
            first_pair_warm_forward_seconds=warm, first_pair_warm_mean_seconds=warm_mean,
            included='explicit input packing + SDK set/run/get/copy/release + official CPU pose codec',
            excluded=['load/init','attribute/performance queries','finite checks','comparisons','hashing','saving'],
            profile_only_diagnostic=True, statistical_or_online_speed_claim=False)
        result['lifecycle'] = {role:{key:receipts[role][key] for key in ('sdk_run_api_calls','successful_synchronous_call_count','context_destroyed','cleanup_errors','affinity_restored','model_memory_bytes','memory_final')} for role in ('normal','profile')}
        result['identity'] = dict(runtime_sha256=RUNNER_SHA, compiler_sha256=COMPILER_SHA,
            ONNX_sha256=exported['onnx']['sha256'], RKNN_sha256=compiled['output']['sha256'],
            checkpoint_sha256=normal['checkpoint_sha256'], original_DA3_weights=True,
            logical_input_shape=[2,3,252,336], native_input_shape=[2,252,336,3], SDK_input_fmt=1,
            SDK_version=normal['sdk_version'], driver_version=normal['driver_version'])
        result['all_internal_operators_on_NPU'] = all(c['internal_CPU_layers'] == 0 and c['targets'].get('GPU',0) == 0 for c in result['operator_coverage'])
        result['limits'] = ['Passed means saved-artifact/receipt/log consistency, not a passed strict FP32 or task-quality gate.',
            'Profile layer time sums are diagnostic and cannot replace separate normal wall timings.',
            'The SDK uses CPU fallback for the recorded internal CPU operators; official pose decoding also runs on CPU.',
            'Board absolute source-after hashes are checked for receipt consistency; this portable host audit does not reread board-only paths.',
            'Original CPU/ORT/official-source audits are separate evidence. Three fixed pairs do not validate SLAM, ATE/RPE, online queues, sustained timing or a VGGT replacement.']
    except Exception as error:
        failures.append(type(error).__name__+': '+str(error))
    result['checks'] = dict(count=checked_count, failures=failures)
    result['status'] = 'passed' if not failures else 'failed'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('normal-dir','profile-dir','cpu-dir','export-dir','compile-dir','ort-dir',
                 'runtime-source','normal-log','profile-log','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists')
    result = audit(args)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=result['checks'], output=str(args.output), sha256=sha(args.output))))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
