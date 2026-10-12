#!/usr/bin/env python3
"""Board-only static export/FP16 compile of original frame0 FC1/GELU/FC2."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)

def sha(p):
    d = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): d.update(b)
    return d.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert platform.machine() == 'aarch64'
    assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
    assert int(Path('/sys/class/devfreq/fdab0000.npu/cur_freq').read_text()) == 300000000
    os.sched_setaffinity(0, {4, 5, 6, 7})
    a.output.mkdir(parents=True, exist_ok=False)
    r = dict(status='running', script_sha256=sha(__file__), original_model_only=True,
        checkpoint_sha256=CHECKPOINT_SHA, capture_sha256=CAPTURE_SHA, config=CONFIG,
        build=dict(do_quantization=False, dataset=None, auto_hybrid=False),
        input_shape=[106, 1024], output_shape=[106, 1024],
        scope='original aggregator.frame_blocks.0.mlp, flatten both frames and all 53 tokens',
        original_gelu='none / exact Erf', full_model_or_realtime_validated=False,
        clock_writes=False, cleanup_errors=[], calls=[])
    sdk = None
    def write(): (a.output / 'result.json').write_text(json.dumps(r, indent=2, allow_nan=False) + '\n')
    write()
    try:
        import numpy as np
        import onnx
        from onnx import helper as H, numpy_helper as N, TensorProto as T
        c = a.capture / 'capture.json'
        assert sha(c) == CAPTURE_SHA
        capture = json.loads(c.read_text())
        assert capture['checkpoint']['sha256'] == CHECKPOINT_SHA
        assert capture['full_model_forward_completed'] and capture['cleanup_errors'] == []
        block = next(v for v in capture['blocks'] if v['role'] == 'frame' and v['index'] == 0)
        assert block['gelu_approximate'] == 'none'
        wpath = a.capture / block['weights']['file']
        assert sha(wpath) == block['weights']['sha256']
        tensors = []
        r['weights_file_sha256'] = sha(wpath)
        r['original_parameter_raw_sha256'] = {}
        with np.load(wpath, allow_pickle=False) as weights:
            for name in ('mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias'):
                x = weights[name]
                pin = block['parameters'][name]
                assert str(x.dtype) == pin['dtype'] == 'float32' and list(x.shape) == pin['shape']
                assert hashlib.sha256(x.tobytes()).hexdigest() == pin['raw_sha256']
                r['original_parameter_raw_sha256'][name] = pin['raw_sha256']
                tensors.append(N.from_array(np.ascontiguousarray(x.T if name.endswith('weight') else x), name))
        for name, value in [('sqrt2', np.sqrt(2.0)), ('one', 1.0), ('half', 0.5)]:
            tensors.append(N.from_array(np.array(value, dtype=np.float32), name))
        nodes = []
        def op(kind, ins, out): nodes.append(H.make_node(kind, ins, [out], name=out))
        op('MatMul', ['input', 'mlp.fc1.weight'], 'fc1_mm')
        op('Add', ['fc1_mm', 'mlp.fc1.bias'], 'fc1')
        op('Div', ['fc1', 'sqrt2'], 'gelu_scaled_input')
        op('Erf', ['gelu_scaled_input'], 'gelu_erf')
        op('Add', ['gelu_erf', 'one'], 'gelu_plus_one')
        op('Mul', ['gelu_plus_one', 'fc1'], 'gelu_product')
        op('Mul', ['gelu_product', 'half'], 'gelu')
        op('MatMul', ['gelu', 'mlp.fc2.weight'], 'fc2_mm')
        op('Add', ['fc2_mm', 'mlp.fc2.bias'], 'output')
        graph = H.make_graph(nodes, 'original_VGGT_frame0_MLP',
            [H.make_tensor_value_info('input', T.FLOAT, [106, 1024])],
            [H.make_tensor_value_info('output', T.FLOAT, [106, 1024])], tensors)
        model = H.make_model(graph, opset_imports=[H.make_opsetid('', 17)])
        model.ir_version = 8
        onnx.checker.check_model(model, full_check=True)
        graph_path = a.output / 'original-frame0-mlp.onnx'
        graph_path.write_bytes(model.SerializeToString())
        r['onnx'] = dict(file=graph_path.name, sha256=sha(graph_path), bytes=graph_path.stat().st_size)
        del model, graph, tensors
        from rknn.api import RKNN
        r['versions'] = {k: importlib.metadata.version(k) for k in ('rknn-toolkit2', 'onnx', 'numpy')}
        assert r['versions']['rknn-toolkit2'].startswith('2.3.2')
        sdk = RKNN(verbose=True)
        for operation, kwargs in [('config', CONFIG), ('load_onnx', dict(model=str(graph_path))),
                ('build', r['build']), ('export_rknn', dict(export_path=str(a.output / 'original-frame0-mlp.rknn')))]:
            started = time.monotonic()
            code = getattr(sdk, operation)(**kwargs)
            r['calls'].append(dict(operation=operation, returncode=code, seconds=time.monotonic() - started))
            write()
            if code != 0: raise RuntimeError(f'{operation} returned {code}')
        path = a.output / 'original-frame0-mlp.rknn'
        r['rknn'] = dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size)
        assert sha(c) == CAPTURE_SHA and sha(wpath) == block['weights']['sha256']
        r['status'] = 'compiled_original_frame0_mlp_diagnostic'
    except BaseException as e:
        r.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        if sdk is not None:
            try: sdk.release(); r['sdk_released'] = True
            except BaseException as e: r['cleanup_errors'].append(str(e))
        write()
        print(json.dumps({k: r.get(k) for k in ('status', 'error', 'rknn', 'sdk_released', 'cleanup_errors')}), flush=True)

if __name__ == '__main__': main()
