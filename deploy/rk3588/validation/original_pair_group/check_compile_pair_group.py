#!/usr/bin/env python3
"""RK3588-only FP32 two-pair chain check, then separate diagnostic FP16 build."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
CHECKPOINT = {'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0', 'bytes': 5026874952}
INPUT_SHAPE = [2, 53, 1024]
BINDINGS = [dict(index=i, name=f'pair{i // 2}_{role}_output',
    shape=INPUT_SHAPE if role == 'frame' else [1, 106, 1024], dtype='float32')
    for i, role in enumerate(('frame', 'global', 'frame', 'global'))]
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def pin(path, expected):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, 'Pinned file changed: ' + str(path))
    return path


def artifact(root, row):
    root, relative = Path(root).resolve(strict=True), Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
    path = pin(root / relative, row['sha256'])
    require(path.is_relative_to(root) and path.stat().st_size == row['bytes'], 'Artifact path/size differs')
    return path


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def tensor(np, root, row, shape):
    value = np.load(artifact(root, row), allow_pickle=False)
    require(value.dtype == np.float32 and list(value.shape) == shape and value.flags.c_contiguous
        and np.isfinite(value).all(), 'Captured FP32 tensor shape/type/finite differs')
    require(row['shape'] == shape and row['dtype'] == 'float32' and row['tensor_bytes'] == value.nbytes
        and row['raw_sha256'] == hashlib.sha256(memoryview(value).cast('B')).hexdigest(), 'Captured tensor identity differs')
    return value


def save_tensor(np, root, filename, value, binding):
    require(value.dtype == np.float32 and list(value.shape) == binding['shape']
        and value.flags.c_contiguous and np.isfinite(value).all(), 'ORT output contract differs')
    path = root / filename
    np.save(path, value, allow_pickle=False)
    return dict(file=filename, bytes=path.stat().st_size, sha256=sha(path),
        name=binding['name'], shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes,
        raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def compare(np, actual, reference):
    delta, expected = actual.astype(np.float64) - reference.astype(np.float64), reference.astype(np.float64)
    outside = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(expected)))
    return dict(strict_rtol=1e-4, strict_atol=1e-5, strict_pass=outside == 0,
        outside_tolerance=outside, elements=int(actual.size), max_abs=float(np.max(np.abs(delta))),
        relative_l2=float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-30)))


def real_cases(capture_root, calibration_root, parent, watched):
    capture_path = pin(capture_root / 'capture.json', CAPTURE_SHA)
    watched.append(capture_path)
    capture = json.loads(capture_path.read_text())
    require(capture['schema'] == 'v53_original_VGGT_48_block_capture' and capture['checkpoint'] == CHECKPOINT
        and capture['full_model_forward_completed'] is True and capture['original_model_only'] is True
        and capture['cleanup_errors'] == [], 'Original completed v53 full capture required')
    blocks = capture['blocks'][:4]
    require([(r['role'], r['index']) for r in blocks] == [('frame', 0), ('global', 0), ('frame', 1), ('global', 1)], 'Original four-block order differs')
    require(all(blocks[i]['output']['raw_sha256'] == blocks[i + 1]['input']['raw_sha256'] for i in range(3)), 'Original capture chain differs')
    require(all(capture['source_hashes'].get(k) == v for k, v in parent['source_hashes'].items()), 'Original captured sources differ')
    cases = [dict(tag='v53', directory=str(capture_root), capture_manifest_sha256=CAPTURE_SHA,
        input=blocks[0]['input'], original_outputs=[r['output'] for r in blocks],
        source_images=capture['source_images'], input_origin='original full FP32 model frame0 input')]
    if calibration_root is not None:
        calibration_path = pin(calibration_root / 'capture.json', CALIBRATION_SHA)
        watched.append(calibration_path)
        calibration = json.loads(calibration_path.read_text())
        require(calibration['status'] == 'completed_original_FP32_calibration_capture'
            and calibration['checkpoint'] == CHECKPOINT and calibration['image_shape'] == [2, 3, 84, 112]
            and calibration['full_forward_completed'] == 8 and calibration['original_model_only'] is True
            and calibration['NPU_or_cache_installed'] is False and calibration['cleanup_errors'] == [], 'Original v56 calibration capture differs')
        require(all(calibration['source_hashes'].get(k) == v for k, v in parent['source_hashes'].items()), 'Calibration captured sources differ')
        group, sample = calibration['groups']['aggregator'][0], calibration['samples'][0]
        item = group['samples'][0]
        require(group['index'] == 0 and item['sample_id'] == sample['sample_id'] == 'calib00'
            and item['source_pair_indices'] == sample['indices'] == [0, 48]
            and item['origin_batch_shape'] == INPUT_SHAPE and item['origin_batch_row'] is None, 'Real second frame0 input differs')
        require(item['artifact']['raw_sha256'] != cases[0]['input']['raw_sha256'], 'Second input repeats the first')
        cases.append(dict(tag='v56-calib00', directory=str(calibration_root), capture_manifest_sha256=CALIBRATION_SHA,
            input=item['artifact'], original_outputs=None, source_images=sample['images'],
            input_origin='saved original full FP32 model activation; no original four-output Torch reference saved'))
    for case in cases:
        for row in [case['input'], *(case['original_outputs'] or [])]:
            watched.append(artifact(Path(case['directory']), row))
    return cases


def run_check(graph, parents, cases, output, report):
    import numpy as np
    import onnxruntime as ort
    report['versions'] = dict(numpy=np.__version__, onnxruntime=ort.__version__, python=sys.version)
    report['ORT'] = dict(optimization='ORT_DISABLE_ALL', intra_op_num_threads=1,
        inter_op_num_threads=1, providers=['CPUExecutionProvider'], timing_is_not_NPU_performance=True)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    sessions = []
    try:
        for path, output_names in [(graph, [b['name'] for b in BINDINGS]),
                *((path, ['frame_output', 'global_output']) for path in parents)]:
            session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
            sessions.append(session)
            require([v.name for v in session.get_inputs()] == ['input']
                and [v.name for v in session.get_outputs()] == output_names, 'ORT graph ABI differs')
        for case in cases:
            x = tensor(np, Path(case['directory']), case['input'], INPUT_SHAPE)
            case.update(group_outputs=[], chain_outputs=[], checks=[])
            started = time.perf_counter()
            actual = sessions[0].run([b['name'] for b in BINDINGS], {'input': x})
            case['group_ORT_CPU_seconds'] = time.perf_counter() - started
            started = time.perf_counter()
            first = sessions[1].run(['frame_output', 'global_output'], {'input': x})
            bridge = first[1].reshape(INPUT_SHAPE)
            require(bridge.flags.c_contiguous, 'Original C-order global-to-frame bridge differs')
            expected = first + sessions[2].run(['frame_output', 'global_output'], {'input': bridge})
            case['chain_ORT_CPU_seconds'] = time.perf_counter() - started
            for index, (a, b, binding) in enumerate(zip(actual, expected, BINDINGS)):
                case['group_outputs'].append(save_tensor(np, output, f"{case['tag']}-group-{index}.npy", a, binding))
                case['chain_outputs'].append(save_tensor(np, output, f"{case['tag']}-chain-{index}.npy", b, binding))
                check = dict(name=binding['name'], byte_equal_to_two_separate_FP32_ONNX=a.tobytes() == b.tobytes(), versus_separate=compare(np, a, b), versus_original_FP32=None)
                if case['original_outputs'] is not None:
                    reference = tensor(np, Path(case['directory']), case['original_outputs'][index], binding['shape'])
                    check['versus_original_FP32'] = dict(group=compare(np, a, reference), chain=compare(np, b, reference))
                case['checks'].append(check)
            write(output / 'check.json', report)
        report['all_separate_ONNX_checks_byte_equal'] = all(v['byte_equal_to_two_separate_FP32_ONNX'] for c in cases for v in c['checks'])
        report['original_FP32_reference_case_count'] = sum(c['original_outputs'] is not None for c in cases)
        report['original_FP32_gate_passed_where_reference_available'] = all(v['versus_original_FP32'][role]['strict_pass'] for c in cases for v in c['checks'] if v['versus_original_FP32'] is not None for role in ('group', 'chain'))
        report['original_FP32_references_cover_all_cases'] = all(c['original_outputs'] is not None for c in cases)
        require(report['all_separate_ONNX_checks_byte_equal'], 'Group differs from separate FP32 ONNX chain; compilation blocked')
        report['status'] = 'checked_separate_ONNX_equivalence_not_quality_accepted'
    finally:
        sessions.clear()
        session = None
        report['sessions_dropped'] = True


def run_compile(graph, args, report):
    check_path = pin(args.check_report, args.check_report_sha256)
    prior = json.loads(check_path.read_text())
    require(prior['status'] == 'checked_separate_ONNX_equivalence_not_quality_accepted'
        and prior['manifest_sha256'] == args.sha256 and prior['graph_sha256'] == report['graph_sha256']
        and prior['script_sha256'] == report['script_sha256'] and prior['cleanup_errors'] == []
        and prior['source_and_inputs_unchanged'] is True and prior['all_separate_ONNX_checks_byte_equal'] is True
        and len(prior['cases']) in (1, 2) and all(len(c['checks']) == 4 and all(v['byte_equal_to_two_separate_FP32_ONNX'] for v in c['checks']) for c in prior['cases']), 'Completed pinned chain check required before compilation')
    report.update(check_receipt_sha256=args.check_report_sha256, check_receipt=str(check_path), cases=prior['cases'],
        original_FP32_gate_passed_where_reference_available=prior['original_FP32_gate_passed_where_reference_available'],
        original_FP32_references_cover_all_cases=prior['original_FP32_references_cover_all_cases'], configuration=CONFIG, build_options=BUILD)
    version = importlib.metadata.version('rknn-toolkit2')
    require(version.split('+')[0] == '2.3.2', 'RKNN Toolkit2 2.3.2 required')
    from rknn.api import RKNN
    report.update(toolkit_version=version, toolkit_source_sha256=sha(Path(sys.modules[RKNN.__module__].__file__)))
    sdk = None
    try:
        sdk = RKNN(verbose=False, verbose_file=str(args.output_dir / 'sdk.log'))
        for name, call in [('config', lambda: sdk.config(**CONFIG)), ('load_onnx', lambda: sdk.load_onnx(model=str(graph))),
                ('build', lambda: sdk.build(**BUILD)), ('export_rknn', lambda: sdk.export_rknn(str(args.output_dir / 'group.rknn')))]:
            operation = dict(name=name, status='attempted')
            report['operations'].append(operation)
            started = time.perf_counter()
            try:
                code = call()
                operation.update(code=code, status='returned')
                require(code in (None, 0), f'{name} returned {code}')
            finally:
                operation['seconds'] = time.perf_counter() - started
                write(args.output_dir / 'result.json', report)
        model = args.output_dir / 'group.rknn'
        report.update(rknn=dict(file='group.rknn', bytes=model.stat().st_size, sha256=sha(model)), status='compiled_not_quality_accepted')
    finally:
        if sdk is not None:
            try:
                code = sdk.release()
                require(code in (None, 0), f'release returned {code}')
                report['released'] = True
            except BaseException as exc:
                report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
        require(not report['cleanup_errors'], 'SDK cleanup failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'output-dir', 'parent-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--mode', choices=('check', 'compile'), required=True)
    parser.add_argument('--capture-dir', type=Path)
    parser.add_argument('--calibration-dir', type=Path)
    parser.add_argument('--check-report', type=Path)
    parser.add_argument('--check-report-sha256')
    args = parser.parse_args()
    if args.mode == 'check' and args.capture_dir is None:
        parser.error('--capture-dir required for check')
    if args.mode == 'compile' and (args.check_report is None or args.check_report_sha256 is None):
        parser.error('--check-report and --check-report-sha256 required for compile')
    compatible = Path('/proc/device-tree/compatible').read_bytes() if Path('/proc/device-tree/compatible').exists() else b''
    if platform.machine() != 'aarch64' or b'rockchip,rk3588' not in compatible:
        parser.error('Actual ORT/SDK execution is restricted to RK3588')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original_cwd, original_tmp, original_affinity = Path.cwd(), os.environ.get('TMPDIR'), os.sched_getaffinity(0)
    report = dict(schema='v64_original_VGGT_pair0_pair1_group_check_compile', mode=args.mode, status='running',
        script_sha256=sha(__file__), manifest_sha256=args.sha256, original_parent_manifest_sha256=PARENT_SHA,
        candidate_accepted=False, quality_accepted=False, realtime_validated=False, operations=[], cases=[], cleanup_errors=[],
        timing_scope='board CPU ORT reference check or compiler elapsed time; not model execution performance')
    watched, pins, started = [], {}, time.perf_counter()
    try:
        os.sched_setaffinity(0, {4, 5, 6, 7})
        manifest_path, parent_path = pin(args.manifest, args.sha256), pin(args.parent_manifest, PARENT_SHA)
        manifest, parent = json.loads(manifest_path.read_text()), json.loads(parent_path.read_text())
        row = manifest['blocks'][0]
        require(manifest['schema'] == 'v64_original_VGGT_pair0_pair1_group_ONNX_manifest'
            and len(manifest['blocks']) == 1 and manifest['original_parent_manifest_sha256'] == PARENT_SHA
            and parent['schema'] == 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest', 'Original group/parent schema differs')
        require(manifest['checkpoint'] == parent['checkpoint'] == CHECKPOINT and manifest['source_hashes'] == parent['source_hashes']
            and manifest['capture_sources'] == parent['capture_sources'] and manifest['capture_sha256'] == parent['capture_sha256'] == CAPTURE_SHA
            and manifest['image_shape'] == parent['image_shape'] == [2, 3, 84, 112], 'Original model provenance differs')
        require(row['role'] == 'pair_group' and row['index'] == 0 and row['indices'] == [0, 1]
            and row['input_name'] == 'input' and row['input_shape'] == INPUT_SHAPE and row['output_bindings'] == BINDINGS
            and row['parents'] == parent['blocks'][:2] and row['parameter_raw_sha256'] == {str(i): parent['blocks'][i]['parameter_raw_sha256'] for i in range(2)}
            and row['positions'] == {str(i): parent['blocks'][i]['positions'] for i in range(2)}, 'Original pair0/pair1 identity or ABI differs')
        require(manifest['boundary_mode'] == 'fp32' and row['added_reshape_count'] == 1
            and row['added_shape_initializer_count'] == 1 and len(row['parent_preservation_proofs']) == 2
            and all(v['inverse_namespace_recovers_every_node_and_initializer_protobuf_bytewise'] is True for v in row['parent_preservation_proofs'])
            and all(row[k] is True for k in ('all_original_parent_nodes_attributes_and_initializer_payloads_retained',
                'all_72_original_parameter_values_preserved_by_parent_composition', 'original_FP32_parent_graph_math_preserved',
                'only_global0_to_frame1_original_view_added', 'all_four_original_pair_outputs_retained',
                'original_aggregator_concat_must_be_retained')), 'Original graph preservation proof differs')
        graph = artifact(manifest_path.parent, row)
        parents = [artifact(parent_path.parent, r) for r in row['parents']]
        watched.extend([Path(__file__).resolve(), manifest_path, parent_path, graph, *parents])
        report.update(graph_sha256=row['sha256'], input_shape=INPUT_SHAPE, output_bindings=BINDINGS, board=dict(machine=platform.machine(), compatible=compatible.decode(errors='replace')))
        if args.mode == 'check':
            report['cases'] = real_cases(args.capture_dir.resolve(strict=True), args.calibration_dir.resolve(strict=True) if args.calibration_dir else None, parent, watched)
        else:
            args.check_report = pin(args.check_report, args.check_report_sha256)
            watched.append(args.check_report)
        pins = {str(path): sha(path) for path in watched}
        report['source_and_input_hashes'] = pins
        if args.mode == 'check':
            run_check(graph, parents, report['cases'], args.output_dir, report)
        else:
            os.chdir(args.output_dir)
            os.environ['TMPDIR'] = str(args.output_dir)
            run_compile(graph, args, report)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            os.chdir(original_cwd)
            if original_tmp is None:
                os.environ.pop('TMPDIR', None)
            else:
                os.environ['TMPDIR'] = original_tmp
            os.sched_setaffinity(0, original_affinity)
            report.update(environment_restored=Path.cwd() == original_cwd and os.environ.get('TMPDIR') == original_tmp,
                affinity_restored=os.sched_getaffinity(0) == original_affinity)
            after = {str(path): sha(path) for path in watched}
            report.update(source_and_input_hashes_after=after, source_and_inputs_unchanged=bool(pins) and after == pins)
            if pins and after != pins:
                report.update(status='failed', error='Pinned sources/graphs/inputs changed during execution')
        except BaseException as exc:
            report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
            report['status'] = 'failed_cleanup'
        report['seconds'] = time.perf_counter() - started
        write(args.output_dir / 'result.json', report)
        if args.mode == 'check':
            write(args.output_dir / 'check.json', report)
    require(not report['cleanup_errors'] and report['status'] != 'failed', 'Final source/cleanup check failed')
    print(json.dumps({k: report.get(k) for k in ('status', 'manifest_sha256', 'graph_sha256', 'all_separate_ONNX_checks_byte_equal', 'original_FP32_gate_passed_where_reference_available', 'seconds', 'rknn')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
