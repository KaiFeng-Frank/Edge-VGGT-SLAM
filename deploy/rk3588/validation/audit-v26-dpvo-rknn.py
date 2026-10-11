#!/usr/bin/env python3
"""Independently audit saved DPVO RKNN encoder arrays; NumPy only, no inference.

Audit success means receipts, hashes, arithmetic and scope statements agree.
It does not turn a failed FP32 numerical gate into quality/VO validation.
Board source/library before/after hashes are recorded evidence, not files
independently rehashed by this offline process.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

RTOL, ATOL = 1e-4, 1e-5
DRIVER_SHA = '895ac2902a66c68048c03e98f353300c7065e7568047479db4fee2f9a80a33d3'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
CHECKPOINT_SHA = '30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
COMMIT = '0ac95b656d1fda91c271d2a106460d19ad966fc7'
SHAPES = {'fmap': (1, 1, 128, 116, 152), 'imap': (1, 1, 384, 116, 152)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def meta(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def errors(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('Matching FP32 array shapes required')
    if actual.dtype != np.float32 or reference.dtype != np.float32 or not (
            np.isfinite(actual).all() and np.isfinite(reference).all()):
        raise ValueError('Finite FP32 array values required')
    a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    diff = a - b
    outside = np.abs(diff) > ATOL + RTOL * np.abs(b)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return dict(rtol=RTOL, atol=ATOL, failed_elements=int(outside.sum()), element_count=int(outside.size),
        gate_passed=not bool(outside.any()), max_abs=float(np.max(np.abs(diff))),
        rms=float(np.sqrt(np.mean(diff * diff))),
        relative_l2=float(np.linalg.norm(diff) / norm_b) if norm_b else None,
        cosine=float(np.dot(a / norm_a, b / norm_b)) if norm_a and norm_b else None,
        array_equal=bool(np.array_equal(actual, reference)),
        bitwise_equal=actual.tobytes(order='C') == reference.tobytes(order='C'),
        finite=True, actual_all_zero=bool(np.all(actual == 0)))


def parse_profile(text):
    rows = []
    for line in text.splitlines():
        if not re.match(r'^\d+\s', line):
            continue
        parts = line.split()
        if len(parts) < 11 or parts[3] not in ('NPU', 'CPU', 'GPU') or not re.fullmatch(r'\d+/\d+/\d+', parts[6]):
            raise ValueError('Unrecognized actual operator profile row')
        workload = [part for part in parts[8:] if re.fullmatch(r'\d+(?:\.\d+)?%/\d+(?:\.\d+)?%/\d+(?:\.\d+)?%', part)]
        if len(workload) != 1:
            raise ValueError('One actual three-core workload field required')
        rows.append(dict(index=int(parts[0]), type=parts[1], dtype=parts[2], target=parts[3],
            input_shapes=parts[4], output_shape=parts[5], elapsed_us=int(parts[7]),
            core_workload_percent=[float(value.rstrip('%')) for value in workload[0].split('/')],
            name=parts[-1]))
    if not rows or [row['index'] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Missing, duplicate or out-of-order operator profile rows')
    total = re.search(r'Total Operator Elapsed Per Frame Time\(us\):\s*(\d+)', text)
    if total is None or sum(row['elapsed_us'] for row in rows) != int(total[1]):
        raise ValueError('Actual per-operator profile elapsed sum differs from reported total')
    return rows


def audit(args):
    checks = []
    result = dict(schema='v26_independent_saved_RKNN_encoder_review', status='failed', checks=checks,
        analysis_script_sha256=sha(__file__), scope=dict(saved_arrays_only=True, images_decoded=False,
            model_forward_run=False, ORT_run=False, board_tasks_run=False,
            board_source_or_library_rehashed_on_host=False, complete_VO_or_realtime_validated=False),
        reported_float_metric_comparison_tolerance=dict(relative=1e-10, absolute=1e-14,
            purpose='Cross-architecture FP64 norm reduction only; FP32 gate, failure counts and bytes stay exact'),
        limitations=['Different network from VGGT; only two original DPVO CNNs, including /4',
            'No correlation, learned update, bundle adjustment, poses or full VO accuracy result',
            'Fixed three inputs and short repeats; no thermal-controlled or sustained performance gate',
            'Normal SDK timing includes native input packing, owned output copies and release',
            'Profile timing is a separate diagnostic run; five internal CPU transposes make this a hybrid path',
            'Successful audit is not successful strict FP32 numerical validation'])

    def check(ok, name):
        checks.append(dict(name=name, passed=bool(ok)))
        if not ok:
            raise ValueError(name)

    def receipt(mode, directory, name):
        path = directory / name
        check(sha(path) == getattr(args, mode + '_sha256'), mode + ' receipt SHA')
        value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        result.setdefault('receipts', {})[mode] = dict(file=path.name, sha256=sha(path))
        return value

    def artifact(directory, item):
        check(isinstance(item['file'], str) and Path(item['file']).name == item['file'], 'simple artifact filename')
        path = directory / item['file']
        check(path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
              and sha(path) == item['sha256'], 'artifact size/file SHA: ' + path.name)
        return path

    def load_arrays(directory, row):
        path = artifact(directory, row['artifact'])
        with np.load(path, allow_pickle=False) as saved:
            check(set(saved.files) == set(SHAPES), 'exact feature keys: ' + path.name)
            values = {name: saved[name].copy() for name in SHAPES}
        for name, value in values.items():
            check(value.dtype == np.float32 and value.shape == SHAPES[name] and np.isfinite(value).all()
                  and np.any(value != 0) and meta(value) == row['arrays'][name],
                  'feature shape/finite/nonzero/raw SHA: ' + path.name + ':' + name)
        return values

    def equal_report(actual, saved, name):
        check(actual.keys() == saved.keys(), name + ' comparison fields')
        for key, value in actual.items():
            same = (math.isclose(value, saved[key], rel_tol=1e-10, abs_tol=1e-14)
                    if key not in ('rtol', 'atol') and type(value) is float and type(saved[key]) in (float, int)
                    else value == saved[key])
            check(same, name + ':' + key)

    try:
        capture = receipt('capture', args.capture_dir, 'capture.json')
        exported = receipt('export', args.export_dir, 'export.json')
        ort = receipt('ort', args.ort_dir, 'ort.json')
        compiled = receipt('compile', args.compile_dir, 'compile.json')
        runs = {mode: receipt(mode, getattr(args, mode + '_dir'), 'run.json') for mode in ('normal', 'profile')}
        check(capture['status'] == 'completed_encoder_cpu_probe_not_VO'
              and exported['status'] == 'exported_not_runtime_validated'
              and ort['status'] == 'completed_CPU_ORT_gate' and ort['exact_reference_gate_passed'] is True,
              'CPU, export and independent ORT prerequisites succeeded')
        for mode, receipt_row in (('capture', capture), ('export', exported), ('ort', ort)):
            loader = receipt_row['loader']
            state = loader['load_state']
            check(receipt_row['state_unchanged'] is True and receipt_row['sources_and_inputs_unchanged'] is True
                  and receipt_row['source_hashes_after'] == receipt_row['source_pins']
                  and receipt_row['cleanup_errors'] == [] and receipt_row['affinity_restored'] is True,
                  mode + ' original state/source/cleanup evidence')
            check(receipt_row['checkpoint_sha256'] == CHECKPOINT_SHA and loader['commit'] == COMMIT
                  and state['strict'] is True and state['inference_state_keys'] == 94
                  and state['inference_parameters'] == 3384324 and state['all_loaded_parameter_bytes_equal'] is True
                  and receipt_row['state_before'] == capture['state_before'], mode + ' full original inference weights binding')
        check(exported['capture_sha256'] == ort['capture_sha256'] == args.capture_sha256
              and ort['export_sha256'] == args.export_sha256 and ort['onnx'] == exported['onnx'], 'CPU/ONNX/ORT receipt lineage')
        artifact(args.export_dir, exported['onnx'])
        check(compiled['status'] == 'compiled_not_runtime_validated' and compiled['script_sha256'] == COMPILER_SHA
              and compiled['configuration'] == dict(target_platform='rk3588', float_dtype='float16',
                  optimization_level=3, do_quantization=False) and compiled['released'] is True
              and not compiled.get('release_error') and compiled['input']['sha256'] == exported['onnx']['sha256'],
              'Exact compiler/source/configuration/release')
        check([row['name'] for row in compiled['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
              and all(row['code'] in (0, None) for row in compiled['operations']), 'compiler operations returned success')
        model = args.compile_dir / Path(compiled['output']['path']).name
        check(model.stat().st_size == compiled['output']['size'] and sha(model) == compiled['output']['sha256'],
              'compiled RKNN file size and SHA')
        result['compiled_model'] = dict(bytes=compiled['output']['size'], sha256=sha(model))
        check(len(capture['inputs']) == len(ort['pairs']) == 3, 'three complete source frames')
        check([row['frame_index'] for row in capture['calls']] == [0, 0, 0, 0, 1, 2], 'original CPU call schedule')
        references, ort_refs, cpu_repeat_hashes = {}, {}, {}
        for i, (seed, ort_row) in enumerate(zip(capture['inputs'], ort['pairs'])):
            check(seed['frame_index'] == ort_row['frame_index'] == i and seed['frame_id'] == seed['source']['frame_id'],
                  'original input frame index and exact timestamp')
            value = np.load(artifact(args.capture_dir, seed['input']), allow_pickle=False)
            check(value.dtype == np.float32 and value.shape == (1, 1, 3, 464, 608)
                  and np.isfinite(value).all() and -.5 <= np.min(value) <= np.max(value) <= 1.5
                  and meta(value) == seed['array'], 'normalized original BGR input raw SHA')
            for cpu in [row for row in capture['calls'] if row['frame_index'] == i]:
                check(cpu['frame_id'] == seed['frame_id'], 'original CPU feature source identity')
                values = load_arrays(args.capture_dir, cpu)
                hashes = {name: meta(a)['tensor_sha256'] for name, a in values.items()}
                if i in cpu_repeat_hashes:
                    check(hashes == cpu_repeat_hashes[i], 'repeated original CPU bytes equal')
                cpu_repeat_hashes[i] = hashes
                references[i] = values
            ort_refs[i] = load_arrays(args.ort_dir, ort_row)
            for name in SHAPES:
                calc = errors(ort_refs[i][name], references[i][name])
                check(calc['gate_passed'] and ort_row['checks'][name]['gate_passed']
                      and ort_row['checks'][name]['failed_elements'] == calc['failed_elements'], 'recomputed saved ORT strict FP32 gate')
        profile_summaries, call_summaries, stable = [], [], {}
        for mode, run in runs.items():
            expected_sequence = [0, 0, 0, 0, 1, 2] if mode == 'normal' else [0, 1, 2]
            check(run['status'] == 'completed_encoder_RKNN_probe_not_VO' and run['script_sha256'] == DRIVER_SHA
                  and run['official_commit'] == COMMIT and run['checkpoint_sha256'] == CHECKPOINT_SHA
                  and run['inputs'] == capture['inputs'] and run['preprocessing'] == capture['preprocessing'],
                  mode + ' original source, preprocessing and frozen runtime binding')
            check(run['original_DPVO_weights'] is True and run['original_VGGT_weights'] is False
                  and run['different_model_from_VGGT'] is True and run['encoder_only'] is True
                  and not any(run[key] for key in ('ORB_used', 'production_changed', 'complete_VO_executed',
                                                  'realtime_validated', 'quality_validated', 'all_operators_on_npu_verified')),
                  mode + ' different network and isolated hybrid CNN scope is explicit')
            check(all(run[key + '_sha256'] == result['receipts'][key]['sha256']
                      for key in ('capture', 'export', 'ort', 'compile')), mode + ' exact receipt lineage')
            check(run['compiled_model_sha256'] == compiled['output']['sha256']
                  and run['compiled_model_bytes'] == compiled['output']['size'], mode + ' compiled artifact identity')
            check(run['cleanup_errors'] == [] and run['context_destroyed'] is True and run['affinity_restored'] is True
                  and run['sources_and_inputs_unchanged'] is True and run['source_hashes_after'] == run['source_pins'],
                  mode + ' context/affinity/source cleanup evidence')
            check({DRIVER_SHA, ABI_SHA, LIBRARY_SHA, HEADER_SHA} <= set(run['source_pins'].values()),
                  mode + ' saved runtime/ABI/header/library pins')
            check(run['sdk_version'].startswith('2.3.2') and run['core_mask'] == 7 and run['cpu_affinity'] == [4, 5, 6, 7],
                  mode + ' measured SDK/core/CPU configuration')
            check(run['collect_perf_detail'] is (mode == 'profile') and run['init_flags'] == (8 if mode == 'profile' else 0)
                  and [row['frame_index'] for row in run['calls']] == expected_sequence
                  and run['sdk_run_api_calls'] == run['successful_synchronous_call_count'] == len(expected_sequence),
                  mode + ' normal/profile flags and actual call count')
            descriptor = run['input_attr']
            check(descriptor['index'] == 0 and descriptor['name'] == 'normalized_bgr' and descriptor['format'] == 1
                  and descriptor['shape'] == [1, 464, 608, 3] and descriptor['type'] == 1
                  and descriptor['quantization_type'] == 2 and descriptor['scale'] == 1 and descriptor['zero_point'] == 0,
                  mode + ' actual native NHWC FP16 input descriptor')
            for i, (name, shape) in enumerate(SHAPES.items()):
                descriptor = run['output_attrs'][i]
                check(descriptor['index'] == i and descriptor['name'] == name and descriptor['format'] == 0
                      and descriptor['shape'] == [shape[0], *shape[2:]] and descriptor['type'] == 1
                      and descriptor['n_elems'] == math.prod(shape) and descriptor['size'] == math.prod(shape) * 2
                      and descriptor['quantization_type'] == 2 and descriptor['scale'] == 1 and descriptor['zero_point'] == 0,
                      mode + ' exact native NCHW output descriptor: ' + name)
            all_gates = []
            for index, row in enumerate(run['calls']):
                i = row['frame_index']
                seed = capture['inputs'][i]
                check(row['call_index'] == index and row['frame_id'] == seed['frame_id']
                      and row['input'] == seed['input'] and row['input_array'] == seed['array'], mode + ' fresh source identity per call')
                check(row['synchronous'] is True and row['output_buffers_released'] is True
                      and row['input_api_format'] == 1 and row['input_pass_through'] == 0
                      and row['input_buffer_shape'] == [1, 464, 608, 3]
                      and row['output_api_want_float'] == 1 and row['output_bytes'] == [9027584, 27082752],
                      mode + ' explicit native packing and SDK-owned output release')
                time_keys = ('input_packing_seconds', 'input_set_seconds', 'run_api_seconds',
                             'outputs_get_seconds', 'output_copy_seconds', 'outputs_release_seconds')
                check(all(math.isfinite(row[key]) and row[key] >= 0 for key in time_keys)
                      and row['forward_seconds'] == row['sdk_io_seconds']
                      and row['forward_seconds'] >= sum(row[key] for key in time_keys) - 1e-6,
                      mode + ' actual forward includes copy/release/layout packing')
                values = load_arrays(getattr(args, mode + '_dir'), row)
                metrics = {}
                for name, a in values.items():
                    digest = meta(a)['tensor_sha256']
                    if (i, name) in stable:
                        check(stable[(i, name)] == digest, 'same input repeated normal/profile output bytes equal')
                    stable[(i, name)] = digest
                    cpu_error, ort_error = errors(a, references[i][name]), errors(a, ort_refs[i][name])
                    equal_report(cpu_error, row['versus_CPU_reference'][name], mode + ' independently recomputed CPU error:' + name)
                    equal_report(ort_error, row['versus_CPU_ORT'][name], mode + ' independently recomputed ORT error:' + name)
                    all_gates.append(cpu_error['gate_passed'])
                    metrics[name] = cpu_error
                call_summaries.append(dict(mode=mode, call_index=index, frame_index=i, frame_id=row['frame_id'],
                    artifact=row['artifact'], arrays=row['arrays'], forward_seconds=row['forward_seconds'], versus_CPU_reference=metrics))
                if mode == 'profile':
                    detail = row['perf_detail']
                    raw = detail['raw_text'].encode('utf-8')
                    check(len(raw) == detail['bytes'] and hashlib.sha256(raw).hexdigest() == detail['raw_sha256'],
                          'actual SDK profile raw bytes/hash')
                    ops = parse_profile(detail['raw_text'])
                    io = [op for op in ops if op['type'] in ('InputOperator', 'OutputOperator')]
                    cpu = [op for op in ops if op['target'] == 'CPU' and op not in io]
                    npu = [op for op in ops if op['target'] == 'NPU']
                    check(len(ops) == 90 and len(io) == 3 and all(op['target'] == 'CPU' for op in io)
                          and len(cpu) == 5 and all(op['type'] == 'Transpose' for op in cpu) and len(npu) == 82,
                          'actual 90 profile rows: 82 internal NPU, 5 internal CPU transpose, 3 CPU IO')
                    profile_summaries.append(dict(frame_index=i, raw_sha256=detail['raw_sha256'], operator_rows=len(ops),
                        internal_NPU_rows=len(npu), internal_CPU_rows=len(cpu), CPU_IO_rows=len(io),
                        internal_CPU_transpose_elapsed_us=sum(op['elapsed_us'] for op in cpu),
                        CPU_IO_elapsed_us=sum(op['elapsed_us'] for op in io), NPU_elapsed_us=sum(op['elapsed_us'] for op in npu),
                        profile_operator_elapsed_us=sum(op['elapsed_us'] for op in ops), CPU_transposes=cpu,
                        NPU_core_workload_patterns=sorted({tuple(op['core_workload_percent']) for op in npu}),
                        all_internal_ops_NPU=False))
                else:
                    check('perf_detail' not in row, 'normal timing has no collected profiling table')
            check(run['exact_FP32_reference_gate_passed'] == all(all_gates), mode + ' overall numerical gate faithfully recorded')
        for mode in runs:
            log_path = getattr(args, mode + '_log') or (getattr(args, mode + '_dir').parent /
                        (getattr(args, mode + '_dir').name + '.log'))
            if log_path.is_file():
                text = log_path.read_text(errors='replace')
                bad = [line for line in text.splitlines() if re.match(r'^\s*E\s+RKNN', line)]
                check(not bad, mode + ' actual log has no E RKNN errors')
                result.setdefault('SDK_logs', {})[mode] = dict(file=log_path.name, sha256=sha(log_path), E_RKNN_lines=bad)
            else:
                result.setdefault('SDK_logs', {})[mode] = dict(checked=False, reason='Companion log not supplied or present')
        warm = [row['forward_seconds'] for row in runs['normal']['calls'][1:4]]
        check(warm == runs['normal']['first_frame_warm_forward_seconds']
              and math.isclose(float(np.mean(warm)), runs['normal']['first_frame_warm_mean_seconds'], abs_tol=1e-15),
              'normal three warm repeats summary')
        result.update(status='passed', calls=call_summaries, profile=profile_summaries,
            verified_NPU_feature_arrays=18, same_frame_repeat_and_normal_profile_bytes_equal=True,
            strict_FP32_gate_passed=all(row['versus_CPU_reference'][name]['gate_passed'] for row in call_summaries for name in SHAPES),
            numerical_scope='Finite nonzero original learned encoders executed on a hybrid NPU/CPU path; strict FP32 gate fails',
            timing=dict(normal_all_six_seconds=[row['forward_seconds'] for row in runs['normal']['calls']],
                first_frame_warm_seconds=warm, first_frame_warm_mean_seconds=float(np.mean(warm)),
                original_CPU_first_frame_warm_mean_seconds=capture['first_frame_warm_mean_seconds'],
                profile_excluded_from_normal_summary=True, output_copy_bytes_per_call=36110336),
            cleanup={mode: dict(context_destroyed=run['context_destroyed'], affinity_restored=run['affinity_restored'],
                               errors=run['cleanup_errors']) for mode, run in runs.items()})
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
    result['check_count'] = len(checks)
    result['failed_check_count'] = sum(not row['passed'] for row in checks)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ('capture', 'export', 'ort', 'compile', 'normal', 'profile'):
        parser.add_argument('--' + mode + '-dir', type=Path, required=True)
        parser.add_argument('--' + mode + '-sha256', required=True)
    for mode in ('normal', 'profile'):
        parser.add_argument('--' + mode + '-log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Audit output already exists; choose a new file')
    result = audit(args)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], checks=result['check_count'], failures=result['failed_check_count'],
                         receipt_sha256=sha(args.output), error=result.get('error'))))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
