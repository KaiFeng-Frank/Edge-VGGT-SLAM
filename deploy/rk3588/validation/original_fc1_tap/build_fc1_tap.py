#!/usr/bin/env python3
"""Board-only single original pair00: append two pre-GELU outputs, check, compile."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import time

BASELINE_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CHECKPOINT = {'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0', 'bytes': 5026874952}
BINDINGS = [
    {'index': 0, 'name': 'frame_output', 'shape': [2, 53, 1024], 'dtype': 'float32'},
    {'index': 1, 'name': 'global_output', 'shape': [1, 106, 1024], 'dtype': 'float32'},
    {'index': 2, 'name': 'frame0/mlp.fc1', 'shape': [2, 53, 4096], 'dtype': 'float32'},
    {'index': 3, 'name': 'global0/mlp.fc1', 'shape': [1, 106, 4096], 'dtype': 'float32'},
]
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''): h.update(data)
    return h.hexdigest()


def pin(path, expected):
    path = Path(path).resolve(strict=True)
    if sha(path) != expected: raise ValueError('Pinned file changed: ' + str(path))
    return path


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def artifact(root, row):
    relative = Path(row['file'])
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Relative owned artifact required')
    path = pin(root / relative, row['sha256'])
    if not path.is_relative_to(root) or path.stat().st_size != row['bytes']: raise ValueError('Artifact size/path differs')
    return path


def record(root, path):
    return dict(file=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-manifest', type=Path, required=True)
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    compatible = Path('/proc/device-tree/compatible').read_bytes()
    if platform.machine() != 'aarch64' or b'rockchip,rk3588' not in compatible:
        p.error('All construction/ORT/SDK execution is restricted to RK3588')
    baseline_path = pin(args.baseline_manifest, BASELINE_SHA)
    baseline = json.loads(baseline_path.read_text())
    if baseline['schema'] != 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest' or baseline['checkpoint'] != CHECKPOINT:
        raise ValueError('Exact original v55 manifest required')
    parent = baseline['blocks'][0]
    if (parent['role'] != 'pair' or parent['index'] != 0 or parent['sha256'] != GRAPH_SHA
            or parent['input_name'] != 'input' or parent['input_shape'] != [2, 53, 1024]
            or parent['output_bindings'] != BINDINGS[:2]): raise ValueError('Original pair00 ABI differs')
    graph_path = artifact(baseline_path.parent, parent)
    capture_root = args.capture_dir.resolve(strict=True)
    capture_path = pin(capture_root / 'capture.json', CAPTURE_SHA)
    capture = json.loads(capture_path.read_text())
    if (capture['schema'] != 'v53_original_VGGT_48_block_capture' or capture['checkpoint'] != CHECKPOINT
            or capture['full_model_forward_completed'] is not True or capture['cleanup_errors'] != []):
        raise ValueError('Completed original full-forward capture required')
    frame, global_ = capture['blocks'][:2]
    if [(r['role'], r['index']) for r in (frame, global_)] != [('frame', 0), ('global', 0)]:
        raise ValueError('Original frame0/global0 capture order differs')
    if frame['output']['raw_sha256'] != global_['input']['raw_sha256']:
        raise ValueError('Original global0 reference must follow actual original frame0 output')
    items = [frame['input'], frame['output'], global_['output']]
    paths = [artifact(capture_root, item) for item in items]
    watched = [Path(__file__).resolve(), baseline_path, graph_path, capture_path, *paths]
    pins = {str(path): sha(path) for path in watched}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v61_original_VGGT_pair00_FC1_tap_build', status='running',
        script_sha256=sha(__file__), board=dict(machine=platform.machine(), device_tree_compatible=compatible.decode(errors='replace')),
        original_baseline_manifest_sha256=BASELINE_SHA, original_pair_onnx_sha256=GRAPH_SHA,
        capture_manifest_sha256=CAPTURE_SHA, capture_directory=str(capture_root),
        source_and_input_hashes=pins, candidate_accepted=False, quality_accepted=False, realtime_validated=False)
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        report['versions'] = dict(numpy=np.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__, python=sys.version)
        def load(item, path):
            a = np.load(path, allow_pickle=False)
            if (str(a.dtype) != 'float32' or list(a.shape) != item['shape'] or a.nbytes != item['tensor_bytes']
                    or not a.flags.c_contiguous or not np.isfinite(a).all()
                    or hashlib.sha256(memoryview(a).cast('B')).hexdigest() != item['raw_sha256']):
                raise ValueError('Original captured tensor differs')
            return a
        x, frame_ref, global_ref = [load(item, path) for item, path in zip(items, paths)]
        if list(x.shape) != [2, 53, 1024]: raise ValueError('Original pair input shape differs')
        raw = graph_path.read_bytes()
        original = onnx.load_model_from_string(raw)
        if original.SerializeToString() != raw: raise ValueError('Original protobuf cannot be proven file-byte exact')
        if [v.name for v in original.graph.output] != [b['name'] for b in BINDINGS[:2]]:
            raise ValueError('Original graph output order differs')
        candidate = copy.deepcopy(original)
        info = {v.name: v for v in candidate.graph.value_info}
        for binding in BINDINGS[2:]:
            value = info.get(binding['name'])
            producer = [n for n in candidate.graph.node if binding['name'] in n.output]
            if (value is None or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                    or [d.dim_value for d in value.type.tensor_type.shape.dim] != binding['shape']
                    or len(producer) != 1 or producer[0].op_type != 'Add'
                    or list(producer[0].input) != [binding['name'] + '.matmul', binding['name'] + '.bias']):
                raise ValueError('Exact original FC1 post-bias/pre-GELU tensor required')
            candidate.graph.output.append(onnx.helper.make_tensor_value_info(binding['name'], onnx.TensorProto.FLOAT, binding['shape']))
        recovered = copy.deepcopy(candidate)
        del recovered.graph.output[2:]
        if recovered.SerializeToString() != raw: raise ValueError('Removing taps must recover exact original protobuf')
        if ([n.SerializeToString() for n in candidate.graph.node] != [n.SerializeToString() for n in original.graph.node]
                or [v.SerializeToString() for v in candidate.graph.initializer] != [v.SerializeToString() for v in original.graph.initializer]):
            raise ValueError('Original nodes/initializers changed')
        onnx.checker.check_model(candidate, full_check=True)
        candidate_path = output / 'pair-00-tap.onnx'
        candidate_path.write_bytes(candidate.SerializeToString())
        row = copy.deepcopy(parent)
        row.pop('rknn', None)
        row.update(record(output, candidate_path), output_bindings=BINDINGS,
            only_two_pre_GELU_outputs_appended=True,
            removing_two_appended_outputs_recovers_original_protobuf_bytes=True,
            all_original_nodes_and_initializers_bytewise_unchanged=True)
        manifest = dict(schema='v61_original_VGGT_pair00_FC1_tap_ONNX_manifest', status='exported_not_runtime_validated',
            checkpoint=CHECKPOINT, image_shape=[2, 3, 84, 112], blocks=[row], script_sha256=sha(__file__),
            original_baseline_manifest_sha256=BASELINE_SHA, original_pair_onnx_sha256=GRAPH_SHA,
            capture_manifest_sha256=CAPTURE_SHA, capture_directory=str(capture_root), candidate_accepted=False)
        write(output / 'export.json', manifest)
        report['export_manifest_sha256'] = sha(output / 'export.json')
        del original, candidate, recovered, raw

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        checks = dict(optimization='ORT_DISABLE_ALL', intra_op_num_threads=1, inter_op_num_threads=1,
            providers=['CPUExecutionProvider'], baseline=[], candidate=[], first_two_outputs_bytewise_equal=False,
            rtol=1e-4, atol=1e-5, candidate_accepted=False)
        def compare(a, b):
            delta = a.astype(np.float64) - b.astype(np.float64)
            outside = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(b.astype(np.float64))))
            return dict(strict_pass=outside == 0, outside_tolerance=outside, elements=int(a.size),
                max_abs=float(np.max(np.abs(delta))), relative_l2=float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(b.astype(np.float64).ravel()), 1e-30)))
        values = []
        for label, path, bindings in (('baseline', graph_path, BINDINGS[:2]), ('candidate', candidate_path, BINDINGS)):
            session = None
            try:
                start = time.perf_counter()
                session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
                init_seconds = time.perf_counter() - start
                if ([i.name for i in session.get_inputs()] != ['input']
                        or [o.name for o in session.get_outputs()] != [b['name'] for b in bindings]):
                    raise ValueError('ORT graph ABI differs')
                start = time.perf_counter()
                arrays = session.run([b['name'] for b in bindings], {'input': x})
                checks[label + '_run_seconds'] = time.perf_counter() - start
                checks[label + '_init_seconds'] = init_seconds
                for index, (a, binding) in enumerate(zip(arrays, bindings)):
                    if a.dtype != np.float32 or list(a.shape) != binding['shape'] or not np.isfinite(a).all():
                        raise ValueError('ORT output shape/type/finite contract differs')
                    saved = output / f'{label}-{index}.npy'
                    np.save(saved, a, allow_pickle=False)
                    result = dict(record(output, saved), name=binding['name'], shape=list(a.shape), dtype=str(a.dtype))
                    if index < 2: result['versus_original_FP32'] = compare(a, (frame_ref, global_ref)[index])
                    checks[label].append(result)
                values.append(arrays[:2])
            finally:
                session = None
        checks['first_two_outputs_bytewise_equal'] = all(a.tobytes() == b.tobytes() for a, b in zip(*values))
        checks['sessions_dropped'] = True
        strict = all(r['versus_original_FP32']['strict_pass'] for label in ('baseline', 'candidate') for r in checks[label][:2])
        checks['strict_original_FP32_passed'] = strict
        checks['status'] = ('completed_bitwise_equal' if strict else 'completed_bitwise_equal_numerical_rejected') if checks['first_two_outputs_bytewise_equal'] else 'failed_bitwise_equivalence'
        write(output / 'check.json', checks)
        report['check_receipt_sha256'] = sha(output / 'check.json')
        if not checks['first_two_outputs_bytewise_equal']: raise ValueError('ORT original outputs changed; compilation blocked')

        folder = output / 'compiler'
        folder.mkdir(exist_ok=False)
        compile_report = dict(configuration=CONFIG, build_options=BUILD, operations=[], released=False, cleanup_errors=[], status='running')
        old_cwd, old_tmp = Path.cwd(), os.environ.get('TMPDIR')
        api = None
        try:
            os.chdir(folder); os.environ['TMPDIR'] = str(folder)
            version = importlib.metadata.version('rknn-toolkit2')
            if version.split('+')[0] != '2.3.2': raise ValueError('RKNN Toolkit2 2.3.2 required')
            from rknn.api import RKNN
            sdk_source = Path(sys.modules[RKNN.__module__].__file__).resolve()
            compile_report.update(toolkit_version=version, toolkit_source_sha256=sha(sdk_source))
            api = RKNN(verbose=False, verbose_file=str(folder / 'sdk.log'))
            for name, fn in (('config', lambda: api.config(**CONFIG)),
                    ('load_onnx', lambda: api.load_onnx(model=str(candidate_path))),
                    ('build', lambda: api.build(**BUILD)),
                    ('export_rknn', lambda: api.export_rknn(str(folder / 'pair.rknn')))):
                operation = dict(name=name, status='attempted')
                compile_report['operations'].append(operation)
                start = time.perf_counter()
                try:
                    code = fn(); operation.update(code=code, status='returned')
                    if code not in (None, 0): raise RuntimeError(f'{name} returned {code}')
                except BaseException as exc:
                    operation.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    raise
                finally: operation['seconds'] = time.perf_counter() - start
            compile_report.update(status='compiled_not_numerically_accepted', rknn=record(output, folder / 'pair.rknn'))
        except BaseException as exc:
            compile_report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            if api is not None:
                operation = dict(name='release', status='attempted')
                compile_report['operations'].append(operation)
                start = time.perf_counter()
                try:
                    code = api.release(); operation.update(code=code, status='returned')
                    if code not in (None, 0): raise RuntimeError(f'release returned {code}')
                    compile_report['released'] = True
                except BaseException as exc:
                    operation.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    compile_report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
                finally: operation['seconds'] = time.perf_counter() - start
            os.chdir(old_cwd)
            if old_tmp is None: os.environ.pop('TMPDIR', None)
            else: os.environ['TMPDIR'] = old_tmp
            compile_report['environment_restored'] = Path.cwd() == old_cwd and os.environ.get('TMPDIR') == old_tmp
            write(folder / 'compile.json', compile_report)
        if not compile_report['released'] or compile_report['cleanup_errors']: raise RuntimeError('SDK cleanup failed')
        after = {str(path): sha(path) for path in watched}
        if after != pins or sha(candidate_path) != row['sha256']: raise ValueError('Sources/inputs/graph changed')
        row['rknn'] = compile_report['rknn']
        manifest.update(schema='v61_original_VGGT_pair00_FC1_tap_RKNN_manifest', status='compiled_not_numerically_accepted',
            configuration=CONFIG, build_options=BUILD, operations=compile_report['operations'], released=True,
            cleanup_errors=[], environment_restored=compile_report['environment_restored'],
            source_and_input_hashes=pins, source_and_input_hashes_after=after, source_and_inputs_unchanged=True,
            export_manifest_sha256=report['export_manifest_sha256'], check_receipt_sha256=report['check_receipt_sha256'],
            compile_receipt_sha256=sha(folder / 'compile.json'), strict_original_FP32_passed=strict,
            artifacts=[record(output, path) for path in sorted(output.rglob('*')) if path.is_file()])
        write(output / 'compiled.json', manifest)
        report.update(status='compiled_not_numerically_accepted', compiled_manifest_sha256=sha(output / 'compiled.json'))
        print(json.dumps(report, allow_nan=False), flush=True)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(output / 'result.json', report)


if __name__ == '__main__':
    main()
