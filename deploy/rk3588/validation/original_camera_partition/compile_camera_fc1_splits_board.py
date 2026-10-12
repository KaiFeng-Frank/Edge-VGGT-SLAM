#!/usr/bin/env python3
"""Board-only original CameraHead trunk.0 FC1/exact-GELU output partition probe.

One fresh process/directory per --chunks 1, 2, or 4. Input is the real FC1
activation, not the older CameraHead input.npy. No inference runs here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

CAPTURE_SHA = '1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1'
STATE_SHA = '8f4fd6accc1ca0899d86ecb2583659cdfdf3c8807888fdb1c0f827b552a5647c'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
PARAMETERS = {
    'trunk.0.mlp.fc1.weight': ([8192, 2048], 'fb11713df490d28ad5e3691e684556824cb001e7fa52450297650c1ce65e42e5'),
    'trunk.0.mlp.fc1.bias': ([8192], 'b75a509e0e799c5a88912fa94bc9d1590663ceaf1f17defce0fd8043d81ab3a1'),
}
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
              mean_values=None, std_values=None, quant_img_RGB2BGR=False,
              model_pruning=False, remove_reshape=False, quantize_weight=False,
              enable_flash_attention=False)
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


def raw_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def artifact(path):
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def ranges(chunks, width=8192):
    require(chunks in (1, 2, 4) and width % chunks == 0, 'Unsupported output partition')
    step = width // chunks
    return [(start, start + step) for start in range(0, width, step)]


def export_graph(weight, bias, chunks, path):
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper as nh
    tensors = [nh.from_array(np.array(value, dtype=np.float32), name)
               for name, value in [('gelu_sqrt_two', math.sqrt(2)), ('gelu_one', 1.), ('gelu_half', .5)]]
    nodes, outputs, pieces = [], [], []
    for i, (start, stop) in enumerate(ranges(chunks)):
        prefix = '' if chunks == 1 else f'branch{i}/'
        def name(local):
            return prefix + local
        wt = np.ascontiguousarray(weight[start:stop].T)
        bt = np.ascontiguousarray(bias[start:stop])
        tensors.extend([nh.from_array(wt, name('mlp.fc1.weight')),
                        nh.from_array(bt, name('mlp.fc1.bias'))])
        output = 'output' if chunks == 1 else name('gelu')
        # Preserve frozen rank3 exact-GELU operand order; no K reduction split.
        specification = [
            ('MatMul', ['input', name('mlp.fc1.weight')], name('mlp.fc1.matmul')),
            ('Add', [name('mlp.fc1.matmul'), name('mlp.fc1.bias')], name('mlp.fc1')),
            ('Div', [name('mlp.fc1'), 'gelu_sqrt_two'], name('gelu_scaled_input')),
            ('Erf', [name('gelu_scaled_input')], name('gelu_erf')),
            ('Add', [name('gelu_erf'), 'gelu_one'], name('gelu_erf_plus_one')),
            ('Mul', [name('mlp.fc1'), name('gelu_erf_plus_one')], name('gelu_before_half')),
            ('Mul', [name('gelu_before_half'), 'gelu_half'], output),
        ]
        nodes.extend(h.make_node(op, inputs, [out], name=out) for op, inputs, out in specification)
        outputs.append(output)
        pieces.append(dict(start=start, stop=stop, weight_raw_sha256=raw_sha(wt), bias_raw_sha256=raw_sha(bt)))
    if chunks > 1:
        nodes.append(h.make_node('Concat', outputs, ['output'], axis=2, name='output'))
    graph = h.make_graph(nodes, f'original_camera_fc1_gelu_chunks{chunks}',
                         [h.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 2, 2048])],
                         [h.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 2, 8192])], tensors)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], producer_name='original-VGGT-FC1-partition')
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, check_type=True)
    # Prove the exported initializer values reconstruct the original W[N,K]/b[N].
    initializers = {t.name: nh.to_array(t) for t in model.graph.initializer}
    weights, biases = [], []
    for i in range(chunks):
        prefix = '' if chunks == 1 else f'branch{i}/'
        weights.append(initializers[prefix + 'mlp.fc1.weight'].T)
        biases.append(initializers[prefix + 'mlp.fc1.bias'])
    require(raw_sha(np.concatenate(weights, axis=0)) == raw_sha(weight), 'Original FC1 weights changed')
    require(raw_sha(np.concatenate(biases)) == raw_sha(bias), 'Original FC1 biases changed')
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, str(path))
    return dict(ONNX_version=onnx.__version__, branches=pieces, nodes=len(model.graph.node),
                initializers=len(model.graph.initializer), all_original_weight_and_bias_values_preserved=True,
                output_order_preserved=True, full_K_reduction_per_channel=True,
                exact_GELU_operand_order='hidden * (1 + erf(hidden / sqrt_two)); then * half')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, default=Path('/mnt/vggt-data/vggt-slam2/experiments/vggt-v15-camera/capture112-r1'))
    parser.add_argument('--chunks', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Only RK3588 board execution is allowed')
    compatible = Path('/proc/device-tree/compatible').read_bytes()
    require(b'rockchip,rk3588' in compatible, 'RK3588 device-tree required')
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    capture_file, state_file = args.capture_dir.resolve() / 'capture.json', args.capture_dir.resolve() / 'camera-state.npz'
    report = dict(schema='v69_original_CameraHead_FC1_GELU_partition_compile', status='running',
                  chunks=args.chunks, script_sha256=sha(__file__), original_capture_sha256=CAPTURE_SHA,
                  original_camera_state_sha256=STATE_SHA, original_parameter_raw_sha256={k: v[1] for k, v in PARAMETERS.items()},
                  input_name='input', input_shape=[1, 2, 2048], output_name='output', output_shape=[1, 2, 8192],
                  scope='original trunk.0.mlp.fc1 plus exact GELU; only output-channel partition',
                  input_requires_original_FC1_activation=True, calibration_executed=False, model_inference_executed=False,
                  candidate_accepted=False, full_model_or_realtime_validated=False, clock_writes=False,
                  board=dict(machine=platform.machine(), device_tree_compatible=compatible.replace(b'\0', b',').decode()),
                  configuration=CONFIG, build_options=BUILD, operations=[], released=None, cleanup_errors=[])
    start = time.monotonic()
    sdk = None
    cwd, tmpdir = Path.cwd(), os.environ.get('TMPDIR')
    pins = {str(Path(__file__).resolve()): report['script_sha256']}
    def write():
        temporary = root / 'result.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(root / 'result.json')
    write()
    try:
        import numpy as np
        require(sha(capture_file) == CAPTURE_SHA and sha(state_file) == STATE_SHA, 'Original camera capture identity mismatch')
        pins.update({str(capture_file): CAPTURE_SHA, str(state_file): STATE_SHA})
        capture = json.loads(capture_file.read_text())
        require(capture['kind'] == 'original_complete_CameraHead_CPU_capture' and capture['status'] == 'completed', 'Incomplete original capture')
        require(capture['cleanup_errors'] == [] and capture['source_and_inputs_unchanged'], 'Original capture cleanup failed')
        require(capture['checkpoint']['sha256'] == CHECKPOINT_SHA and capture['checkpoint']['bytes'] == 5026874952, 'Wrong original checkpoint')
        state = capture['artifacts']['camera-state.npz']
        require(state['sha256'] == STATE_SHA and state_file.stat().st_size == state['bytes'], 'Original state size mismatch')
        values = []
        with np.load(state_file, allow_pickle=False) as archive:
            for key, (shape, digest) in PARAMETERS.items():
                value = np.ascontiguousarray(archive[key])
                require(list(value.shape) == shape and value.dtype == np.float32 and np.isfinite(value).all(), 'Original parameter type/shape mismatch')
                require(raw_sha(value) == digest and state['tensors'][key]['sha256'] == digest, 'Original parameter values changed')
                values.append(value)
        report['checkpoint'] = capture['checkpoint']
        report['NumPy_version'] = np.__version__
        report['graph_proof'] = export_graph(*values, args.chunks, root / 'fc1-gelu.onnx')
        report['onnx'] = artifact(root / 'fc1-gelu.onnx')
        pins[str(root / 'fc1-gelu.onnx')] = report['onnx']['sha256']
        report['status'] = 'exported_not_runtime_validated'
        write()
        if not args.export_only:
            report['RKNN_Toolkit2_version'] = importlib.metadata.version('rknn-toolkit2')
            require(report['RKNN_Toolkit2_version'].split('+')[0] == '2.3.2', 'RKNN Toolkit2 2.3.2 required')
            os.chdir(root)
            os.environ['TMPDIR'] = str(root)
            from rknn.api import RKNN
            sdk = RKNN(verbose=True, verbose_file=str(root / 'sdk.log'))
            report['released'] = False
            for name, function in (
                ('config', lambda: sdk.config(**CONFIG)),
                ('load_onnx', lambda: sdk.load_onnx(model=str(root / 'fc1-gelu.onnx'))),
                ('build', lambda: sdk.build(**BUILD)),
                ('export_rknn', lambda: sdk.export_rknn(str(root / 'fc1-gelu.rknn'))),
            ):
                operation = dict(name=name, status='running')
                report['operations'].append(operation)
                write()
                begin = time.monotonic()
                try:
                    code = function()
                    operation.update(code=code, seconds=time.monotonic() - begin, status='completed')
                    require(code in (None, 0), f'{name} returned {code}')
                except BaseException as exc:
                    operation.update(status='failed', seconds=time.monotonic() - begin, error=f'{type(exc).__name__}: {exc}')
                    raise
            report['rknn'] = artifact(root / 'fc1-gelu.rknn')
            report['status'] = 'compiled_not_quality_accepted'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        if sdk is not None:
            try:
                code = sdk.release()
                report['release_code'] = code
                report['released'] = code in (None, 0)
                require(report['released'], f'release returned {code}')
            except BaseException as exc:
                report['cleanup_errors'].append(f'release: {type(exc).__name__}: {exc}')
        try:
            os.chdir(cwd)
            if tmpdir is None:
                os.environ.pop('TMPDIR', None)
            else:
                os.environ['TMPDIR'] = tmpdir
            report['environment_restored'] = Path.cwd() == cwd and os.environ.get('TMPDIR') == tmpdir
            report['source_and_inputs_unchanged'] = all(sha(Path(p)) == digest for p, digest in pins.items())
            require(report['environment_restored'] and report['source_and_inputs_unchanged'], 'Environment or input changed')
        except BaseException as exc:
            report['cleanup_errors'].append(f'restore/identity: {type(exc).__name__}: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['seconds'] = time.monotonic() - start
        write()
        print(json.dumps(dict(status=report['status'], chunks=args.chunks, result_sha256=sha(root / 'result.json'))), flush=True)
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
