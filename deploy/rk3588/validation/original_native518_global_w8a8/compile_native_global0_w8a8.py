#!/usr/bin/env python3
"""Board-only held-out original FP32 gate and calibrated whole-global0 W8A8 build.

The original c440 graph is read unchanged. Exactly four representative captured
inputs calibrate; two disjoint saved original Torch outputs gate that graph
before SDK import. A production-ORT --verify-only receipt may be reused in the
SDK interpreter. Quantized quality and performance are never accepted here.
"""
import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
PARENT_SHA = 'c44094793c3b590c6066cbdd634cf0c4d8bd5b75987c8b491db7b17b2201f55d'
PARENT_BYTES = 51469885
WEIGHTS_CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CONFIG_SOURCE_SHA = '98a5442d064481df4da074d9d591ce914206a91e306542276662bfa1539c4b05'
PARENT_CONFIG_SOURCE_SHA = 'ade8f932c267bdc2a9ff5854e305d87456984630025aa34c0c448499f5a6a81b'
RUNTIME_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
SHAPE = [1, 2082, 1024]
SCHEMA = 'v115_original_native518_global0_W8A8_compile'
VERIFY_SUCCESS = 'verified_original_global0_FP32_heldout_not_quantized'
SUCCESS = 'compiled_original_global0_W8A8_not_quality_accepted'
CAPTURE_SCHEMA = 'v115_original_native518_global0_representative_capture'
CAPTURE_SUCCESS = 'completed_original_native518_global0_input_capture'
FIXED_GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
ORIGINAL_CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
CONFIG = dict(ORIGINAL_CONFIG, quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
CALIBRATION_IDS = ['calib00', 'calib01', 'calib02', 'calib03']
EVALUATION_IDS = ['heldout00', 'heldout01']
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'CUDA_VISIBLE_DEVICES', 'TMPDIR')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def raw(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def pin(path, digest, watched, size=None):
    path = Path(path).resolve(strict=True)
    require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest) and
        sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
    watched[str(path)] = digest
    return path


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def artifact(root, row, watched):
    relative = Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Receipt-relative artifact required')
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    return pin(path, row['sha256'], watched, row['bytes'])


def array(root, row, watched):
    import numpy as np
    path = artifact(root, row, watched)
    value = np.load(path, allow_pickle=False)
    require(value.dtype == np.float32 and list(value.shape) == row['shape'] == SHAPE and
        row['dtype'] == 'float32' and value.flags.c_contiguous and np.isfinite(value).all() and
        row['finite'] is True and value.nbytes == row['tensor_bytes'] and raw(value) == row['raw_sha256'],
        'Finite original complete native array differs: ' + str(path))
    return path, value


def compare(actual, reference):
    import numpy as np
    require(actual.shape == reference.shape == tuple(SHAPE) and actual.dtype == reference.dtype == np.float32 and
        np.isfinite(actual).all() and np.isfinite(reference).all(), 'Complete finite FP32 tensors required')
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    diff = a - b
    norm, error = float(np.linalg.norm(b.ravel())), float(np.linalg.norm(diff.ravel()))
    relative = error / norm if norm else (0.0 if not error else 1e300)
    max_abs = float(np.max(np.abs(diff)))
    strict = bool(np.allclose(actual, reference, rtol=FIXED_GATE['rtol'], atol=FIXED_GATE['atol']))
    outside = int(np.count_nonzero(np.abs(diff) > FIXED_GATE['atol'] + FIXED_GATE['rtol'] * np.abs(b)))
    return dict(relative_l2=relative, max_abs=max_abs, strict_allclose=strict, outside=outside,
        byte_equal=actual.tobytes() == reference.tobytes(),
        fixed_gate_passed=relative <= FIXED_GATE['relative_l2_max'] and max_abs <= FIXED_GATE['max_abs_max'] and strict)


def literal_config(path):
    tree = ast.parse(path.read_text())
    values = [node.value for node in tree.body if isinstance(node, ast.Assign) and
              any(isinstance(target, ast.Name) and target.id == 'CONFIG' for target in node.targets)]
    require(len(values) == 1 and isinstance(values[0], ast.Call) and isinstance(values[0].func, ast.Name) and
        values[0].func.id == 'dict' and not values[0].args and all(k.arg for k in values[0].keywords),
        'Frozen literal CONFIG assignment required')
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in values[0].keywords}


def original_graph(path, parameters, report):
    import numpy as np
    import onnx
    from onnx import numpy_helper as N
    model = onnx.load_model(str(path), load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    graph = model.graph
    require(len(graph.node) == 53 and len(graph.initializer) == 44 and
        len(graph.input) == len(graph.output) == 1 and not model.functions and
        any(node.op_type == 'Erf' for node in graph.node), 'Complete original global0 K4 graph required')
    bindings = []
    for value, name in ((graph.input[0], 'input'), (graph.output[0], 'output')):
        require(value.name == name and value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
            [d.dim_value for d in value.type.tensor_type.shape.dim] == SHAPE, 'Original static FP32 IO differs')
        bindings.append(dict(index=0, name=name, shape=SHAPE, dtype='float32'))
    inits = {value.name: value for value in graph.initializer}
    require(len(inits) == 44 and len(parameters) == 18 and
        all(value.data_location != onnx.TensorProto.EXTERNAL for value in graph.initializer), 'Self-contained original graph required')
    linear = {'attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight'}
    for name, meta in parameters.items():
        stored = name + '_transposed' if name in linear else name
        if name == 'mlp.fc2.weight':
            value = np.concatenate([N.to_array(inits[stored + '.part' + str(i)]) for i in range(4)], axis=0).T
        else:
            value = N.to_array(inits[stored])
            if name in linear:
                value = value.T
        value = np.ascontiguousarray(value)
        require(dict(shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
            raw_sha256=raw(value)) == meta and np.isfinite(value).all(), 'Original learned parameter differs: ' + name)
    report.update(input_bindings=[bindings[0]], output_bindings=[bindings[1]],
        original_parameters=parameters, original_parameter_count=18, original_initializer_count=44, original_node_count=53,
        all18_original_learned_parameters_verified=True, original_ONNX_modified=False,
        original_Erf_GELU_FC2K4_RoPE_residuals_and_full_attention_preserved=True)
    del model, graph, inits, value
    gc.collect()


def prepare(a, r, watched, out):
    graph = pin(a.onnx, PARENT_SHA, watched, PARENT_BYTES)
    require(a.onnx_sha256 == PARENT_SHA, 'Frozen original c440 parent required')
    weights = json.loads(pin(a.weights_capture, WEIGHTS_CAPTURE_SHA, watched).read_text())
    require(weights['checkpoint'] == CHECKPOINT and weights['status'] == 'completed_original_FP32_full_forward_and_48_block_capture' and
        weights['full_model_forward_completed'] and not weights['cleanup_errors'], 'Original full-forward parameter lineage required')
    parameters = next(row['parameters'] for row in weights['blocks'] if row['role'] == 'global' and row['index'] == 0)
    cp = pin(a.capture, a.capture_sha256, watched)
    source = pin(a.capture_source, a.capture_source_sha256, watched)
    c = json.loads(cp.read_text())
    clean_flags = ('source_and_inputs_unchanged', 'model_state_unchanged', 'hooks_removed', 'affinity_restored',
        'environment_restored', 'sys_path_restored', 'all_NPU_contexts_closed', 'resource_counts_balanced',
        'calibration_evaluation_disjoint', 'heldout_not_in_calibration', 'quality_and_fixed_PNG_disjoint')
    require(c['schema'] == CAPTURE_SCHEMA and c['status'] == CAPTURE_SUCCESS and
        c['script_sha256'] == a.capture_source_sha256 and c['checkpoint'] == CHECKPOINT and
        c['original_parameters'] == parameters and c['original_CPU_global0_forward_count'] == 6 and
        all(c.get(name) is True for name in clean_flags) and
        not c['cleanup_errors'] and c['calibration_ids'] == CALIBRATION_IDS and c['evaluation_ids'] == EVALUATION_IDS,
        'Clean pinned six-case original native global0 capture required')
    records = c['records']
    require(len(records) == 6 and [row['id'] for row in records] == CALIBRATION_IDS + EVALUATION_IDS and
        [row['role'] for row in records] == ['calibration'] * 4 + ['heldout'] * 2, 'Exactly four calibration and two held-out records required')
    for name, digest in c['source_hashes'].items():
        if name.startswith('third_party/vggt/'):
            if name in weights['source_hashes']:
                require(weights['source_hashes'][name] == digest, 'Original source lineage differs')
            pin(a.source_root / name, digest, watched)
    dataset_paths, dataset_rows = [], []
    for i, row in enumerate(records):
        require(row['source_pair_indices'] and len(row['source_pair_indices']) == len(row['source_images']) == 2 and
            row['preprocessed_pixels']['shape'] == [2, 3, 392, 518], 'Native real two-frame capture required')
        for image in row['source_images']:
            pin(image['file'], image['sha256'], watched, image['bytes'])
        path, value = array(cp.parent, row['input'], watched)
        del value
        ref = row['reference']
        require(ref['shape'] == SHAPE and ref['dtype'] == 'float32' and ref['tensor_bytes'] == 8527872 and
            ref['finite'] is True and ref['saved'] is (i >= 4), 'Original same-input reference identity differs')
        if i < 4:
            require(not any(ch.isspace() for ch in str(path)), 'Calibration path contains whitespace')
            dataset_paths.append(str(path))
            dataset_rows.append(dict(id=row['id'], input_sha256=row['input']['sha256'], path=str(path)))
        else:
            _, reference = array(cp.parent, ref, watched)
            del reference
    require(len({row['input']['raw_sha256'] for row in records}) == 6 and
        not {image['sha256'] for row in records[:4] for image in row['source_images']} &
            {image['sha256'] for row in records[4:] for image in row['source_images']} and
        not {row['preprocessed_pixels']['raw_sha256'] for row in records[:4]} &
            {row['preprocessed_pixels']['raw_sha256'] for row in records[4:]}, 'Calibration and held-out images/tensors must be disjoint')
    captured_dataset = artifact(cp.parent, c['dataset'], watched)
    require(c['dataset']['line_count'] == 4 and c['dataset']['input_count'] == 1 and c['dataset']['paths_are_absolute'] and
        captured_dataset.read_text().splitlines() == dataset_paths, 'Capture dataset must contain exactly four calibration paths')
    dataset = out / 'dataset.txt'
    dataset.write_text(''.join(path + '\n' for path in dataset_paths))
    pin(dataset, sha(dataset), watched)
    for source_path, digest, expected in ((a.quantization_config_source, CONFIG_SOURCE_SHA, CONFIG),
        (a.parent_config_source, PARENT_CONFIG_SOURCE_SHA, ORIGINAL_CONFIG)):
        source_path = pin(source_path, digest, watched)
        require(literal_config(source_path) == expected, 'Frozen compiler configuration differs')
    runtime = pin(a.runtime_library, RUNTIME_SHA, watched)
    original_graph(graph, parameters, r)
    r.update(onnx=dict(file=str(graph), bytes=PARENT_BYTES, sha256=PARENT_SHA), graph_sha256=PARENT_SHA,
        capture=dict(file=str(cp), sha256=a.capture_sha256),
        capture_source=dict(file=str(source), sha256=a.capture_source_sha256),
        original_source_hashes=c['source_hashes'], checkpoint=CHECKPOINT,
        source_weights_capture_sha256=WEIGHTS_CAPTURE_SHA, quantization_config_source_sha256=CONFIG_SOURCE_SHA,
        parent_config_source_sha256=PARENT_CONFIG_SOURCE_SHA, runtime_library=dict(file=str(runtime), sha256=RUNTIME_SHA),
        capture_clean_flags={name: c[name] for name in clean_flags},
        evaluation_ids=EVALUATION_IDS, calibration_evaluation_disjoint=True, heldout_not_in_calibration=True,
        dataset=dict(file=dataset.name, bytes=dataset.stat().st_size, sha256=sha(dataset), line_count=4,
            input_count=1, calibration_ids=CALIBRATION_IDS, rows=dataset_rows), configuration=CONFIG,
        build_options=dict(do_quantization=True, dataset=str(dataset), auto_hybrid=False))
    return graph, cp, records, dataset


def verify(a, r, watched, out, graph, cp, records):
    import numpy as np
    session = None
    prior = None
    if a.verified_result:
        path = pin(a.verified_result, a.verified_result_sha256, watched)
        prior = json.loads(path.read_text())
        require(prior['schema'] == SCHEMA and prior['status'] == VERIFY_SUCCESS and prior['verify_only'] and
            prior['script_sha256'] == r['script_sha256'] and prior['graph_sha256'] == PARENT_SHA and
            prior['capture']['sha256'] == a.capture_sha256 and prior['capture_source']['sha256'] == a.capture_source_sha256 and
            prior['fixed_gate'] == FIXED_GATE and prior['original_FP32_gate_passed'] and prior['session_dropped'] and
            prior['source_and_inputs_unchanged'] and prior['affinity_restored'] and prior['environment_restored'] and
            prior['cwd_restored'] and not prior['cleanup_errors'] and not prior['calibration_executed'] and
            [row['id'] for row in prior['FP32_ONNX_checks']] == EVALUATION_IDS, 'Clean same-source board FP32 gate required')
        r['verified_result'] = dict(file=str(path), sha256=a.verified_result_sha256)
        r['ORT_version'] = prior['ORT_version']
    else:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
        require(session.get_providers() == ['CPUExecutionProvider'] and
            [(v.name, v.shape, v.type) for v in session.get_inputs()] == [('input', SHAPE, 'tensor(float)')] and
            [(v.name, v.shape, v.type) for v in session.get_outputs()] == [('output', SHAPE, 'tensor(float)')], 'Exact board CPU FP32 original graph IO required')
        r.update(ORT_executed_on_board=True, ORT_version=ort.__version__)
    checks = []
    try:
        for i, record in enumerate(records[4:]):
            _, reference = array(cp.parent, record['reference'], watched)
            if prior:
                old = prior['FP32_ONNX_checks'][i]
                require(old['id'] == record['id'] and old['input_sha256'] == record['input']['sha256'] and
                    old['reference_sha256'] == record['reference']['sha256'], 'Saved held-out gate identity differs')
                _, actual = array(path.parent, old['output'], watched)
                result = compare(actual, reference)
                require(set(result) == set(old['comparison']) and all(
                    math.isclose(result[key], old['comparison'][key], rel_tol=1e-11, abs_tol=1e-14)
                    if key in ('relative_l2', 'max_abs') else result[key] == old['comparison'][key]
                    for key in result), 'Saved gate metrics do not recompute')
                entry = dict(old, comparison=result)
                entry['output'] = dict(old['output'], file=str((path.parent / old['output']['file']).resolve()))
            else:
                _, value = array(cp.parent, record['input'], watched)
                tick = time.perf_counter()
                actual, = session.run(['output'], {'input': value})
                seconds = time.perf_counter() - tick
                result = compare(actual, reference)
                file = out / ('fp32-' + record['id'] + '.npy')
                np.save(file, actual, allow_pickle=False)
                meta = dict(file=file.name, bytes=file.stat().st_size, sha256=sha(file), raw_sha256=raw(actual),
                    shape=SHAPE, dtype='float32', tensor_bytes=actual.nbytes, finite=True)
                pin(file, meta['sha256'], watched)
                entry = dict(id=record['id'], input_sha256=record['input']['sha256'],
                    reference_sha256=record['reference']['sha256'], output=meta, comparison=result,
                    board_CPU_ORT_seconds_not_NPU_performance=seconds)
                del value
            checks.append(entry)
            r['FP32_ONNX_checks'] = checks
            atomic(out / 'result.json', r)
            require(result['fixed_gate_passed'], 'Unchanged original held-out FP32 gate failed: ' + record['id'])
            del actual, reference
        r['original_FP32_gate_passed'] = len(checks) == 2 and all(row['comparison']['fixed_gate_passed'] for row in checks)
        r['saved_metadata_float_recompute_tolerance'] = dict(relative_tolerance=1e-11, absolute_tolerance=1e-14,
            scope='Only recomputed numeric receipt agreement across NumPy/BLAS; original fixed tensor gate unchanged')
    finally:
        session = None
        gc.collect()
        r['session_dropped'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', type=Path, required=True); p.add_argument('--onnx-sha256', required=True)
    p.add_argument('--capture', type=Path, required=True); p.add_argument('--capture-sha256', required=True)
    p.add_argument('--capture-source', type=Path, required=True); p.add_argument('--capture-source-sha256', required=True)
    p.add_argument('--source-root', type=Path, default=BASE / 'src')
    p.add_argument('--weights-capture', type=Path, default=BASE / 'experiments/vggt-v53-capture-r1/capture.json')
    p.add_argument('--quantization-config-source', type=Path, required=True)
    p.add_argument('--parent-config-source', type=Path, required=True)
    p.add_argument('--runtime-library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--verified-result', type=Path); p.add_argument('--verified-result-sha256')
    a = p.parse_args()
    require(bool(a.verified_result) == bool(a.verified_result_sha256) and not (a.verify_only and a.verified_result), 'Gate path/hash must be paired; verify-only creates a fresh gate')
    require(platform.machine() == 'aarch64' and sys.byteorder == 'little' and
        b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'Calibration, ORT and SDK execution are RK3588/S-board only')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    watched = {str(Path(__file__).resolve()): sha(__file__)}
    r = dict(schema=SCHEMA, status='running', script_sha256=sha(__file__), verify_only=a.verify_only,
        original_model_only=True, whole_block=True, component_only=True, candidate_accepted=False,
        full_model_trajectory_or_realtime_accepted=False, fixed_gate=FIXED_GATE,
        graph_rewrite_or_GELU_approximation_performed=False, calibration_executed=False,
        ORT_executed_on_board=False, SDK_or_NPU_inference_executed=False, system_or_frequency_writes=False,
        calls=[], FP32_ONNX_checks=[], cleanup_errors=[], sdk_released=False, session_dropped=False)
    oldcwd, oldenv, oldaff = os.getcwd(), dict(os.environ), os.sched_getaffinity(0)
    handlers, sdk = {}, None
    began = time.monotonic()
    def interrupted(number, frame):
        raise InterruptedError('Termination signal ' + str(number))
    atomic(out / 'result.json', r)
    try:
        free = shutil.disk_usage(out).free
        r['storage_preflight'] = dict(output_filesystem_free_bytes=free, required_free_bytes=1 << 30,
            output_cwd_TMPDIR=str(out), original_parent_read_in_place=True, SDK_verbose=False)
        require(free >= 1 << 30, 'Compiler rootfs output needs at least1GiB free; do not delete raw evidence')
        require({4, 6}.issubset(oldaff), 'A76 compiler cores unavailable')
        os.sched_setaffinity(0, {4, 6})
        for name in THREAD_ENV:
            os.environ[name] = '' if name == 'CUDA_VISIBLE_DEVICES' else (str(out) if name == 'TMPDIR' else '1')
        os.chdir(out)
        for number in (signal.SIGINT, signal.SIGTERM):
            handlers[number] = signal.signal(number, interrupted)
        graph, cp, records, dataset = prepare(a, r, watched, out)
        verify(a, r, watched, out, graph, cp, records)
        require(r['original_FP32_gate_passed'] and r['session_dropped'], 'Two fixed FP32 gates must pass before SDK import')
        if a.verify_only:
            r['status'] = VERIFY_SUCCESS
        else:
            r['toolkit_version'] = importlib.metadata.version('rknn-toolkit2')
            require(r['toolkit_version'].split('+')[0] == '2.3.2', 'Pinned RKNN Toolkit2 2.3.2 required')
            from rknn.api import RKNN
            sdk = RKNN(verbose=False, verbose_file=str(out / 'sdk.log'))
            target = out / 'global0-w8a8.rknn'
            operations = [('config', lambda: sdk.config(**CONFIG)),
                ('load_onnx', lambda: sdk.load_onnx(model=str(graph))),
                ('build', lambda: sdk.build(**r['build_options'])),
                ('export_rknn', lambda: sdk.export_rknn(str(target)))]
            for name, operation in operations:
                entry = dict(operation=name, status='running')
                r['calls'].append(entry)
                if name == 'build':
                    r['calibration_build_invoked'] = True
                atomic(out / 'result.json', r)
                tick = time.monotonic()
                code = operation()
                entry.update(status='completed', returncode=code, seconds=time.monotonic() - tick)
                atomic(out / 'result.json', r)
                require(code in (0, None), 'SDK ' + name + ' failed: ' + str(code))
                if name == 'build':
                    r['calibration_executed'] = True
            require(target.stat().st_size > 0, 'Empty RKNN output')
            r.update(status=SUCCESS, all_operations_succeeded=True,
                rknn=dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target)))
    except BaseException as error:
        r.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
    finally:
        if sdk is not None:
            try:
                code = sdk.release()
                r['sdk_release_code'] = code
                require(code in (0, None), 'SDK release failed: ' + str(code))
                r['sdk_released'] = True
            except BaseException as error:
                r['cleanup_errors'].append('SDK release: ' + str(error))
        for name, action in (('cwd', lambda: os.chdir(oldcwd)), ('affinity', lambda: os.sched_setaffinity(0, oldaff)),
            ('environment', lambda: (os.environ.clear(), os.environ.update(oldenv)))):
            try:
                action()
            except BaseException as error:
                r['cleanup_errors'].append(name + ': ' + str(error))
        for number, handler in handlers.items():
            signal.signal(number, handler)
        r.update(cwd_restored=os.getcwd() == oldcwd, affinity_restored=os.sched_getaffinity(0) == oldaff,
            environment_restored=dict(os.environ) == oldenv)
        try:
            r['source_and_inputs_unchanged'] = all(sha(path) == digest for path, digest in watched.items())
            r['pinned_files'] = watched
            r['output_files'] = [dict(file=str(file.relative_to(out)), bytes=file.stat().st_size)
                for file in out.rglob('*') if file.is_file()]
            r['output_filesystem_free_bytes_after'] = shutil.disk_usage(out).free
            log = out / 'sdk.log'
            if log.is_file():
                r['sdk_log'] = dict(file=log.name, bytes=log.stat().st_size, sha256=sha(log))
        except BaseException as error:
            r['source_and_inputs_unchanged'] = False
            r['cleanup_errors'].append('Pins/storage: ' + str(error))
        if not all(r.get(k) for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'cwd_restored')) or r['cleanup_errors']:
            r['status'] = 'failed_cleanup'
        r.update(seconds=time.monotonic() - began, peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        atomic(out / 'result.json', r)
        print(json.dumps({k: r.get(k) for k in ('status', 'error', 'rknn', 'cleanup_errors')}), flush=True)
    return 0 if r['status'] == (VERIFY_SUCCESS if a.verify_only else SUCCESS) else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
