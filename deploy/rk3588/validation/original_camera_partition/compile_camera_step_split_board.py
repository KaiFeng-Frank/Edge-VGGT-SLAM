#!/usr/bin/env python3
"""RK3588-only: split all four original camera-step FC1s along N, retain GELU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

PARENT_SHA = 'f89fb0e66f43a3d3b94a76ec0124d4f521fa1ce9a6c93efab5e362d3a614eb9f'
HELPER_SHA = 'cb107c093cee8359724095a5298d0cdf2407499503a1fa0358da49d2055c2894'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def rewrite(source, destination):
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper as nh
    model = onnx.load(str(source))
    require([(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in model.graph.input]
            == [('camera_tokens', [1, 2, 2048]), ('conditioning_pose', [1, 2, 9])], 'Original step inputs required')
    require([(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in model.graph.output]
            == [('pose_delta', [1, 2, 9])], 'Original delta output required')
    nodes, initializers = list(model.graph.node), list(model.graph.initializer)
    require(len({n.name for n in nodes}) == len(nodes), 'Unique original node names required')
    consumers = {}
    for node in nodes:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    def one_consumer(value, op):
        result = consumers.get(value, [])
        require(len(result) == 1 and result[0].op_type == op, f'Unique {op} consumer required: {value}')
        return result[0]
    removed_nodes, removed_initializers, substitutions, tensors, replacements = set(), set(), {}, [], []
    removed_values = set()
    for index in range(4):
        wkey, bkey = (f'trunk.{index}.mlp.fc1.{suffix}' for suffix in ('weight', 'bias'))
        weights = [t for t in initializers if t.name.endswith(wkey)]
        biases = [t for t in initializers if t.name.endswith(bkey)]
        require(len(weights) == len(biases) == 1, 'Exactly four original FC1 bindings required')
        w, b = weights[0], biases[0]
        wa, ba = nh.to_array(w), nh.to_array(b)
        require(wa.shape == (8192, 2048) and ba.shape == (8192,) and wa.dtype == ba.dtype == np.float32,
                'Original FC1 shapes/dtypes required')
        require(np.isfinite(wa).all() and np.isfinite(ba).all(), 'Finite original FC1 weights required')
        transpose = one_consumer(w.name, 'Transpose')
        require(len(transpose.output) == 1 and list(transpose.input) == [w.name], 'Original weight Transpose mismatch')
        attrs = {a.name: h.get_attribute_value(a) for a in transpose.attribute}
        require(attrs.get('perm') == [1, 0], 'Original FC1 W[N,K] transpose required')
        matmul = one_consumer(transpose.output[0], 'MatMul')
        require(len(matmul.output) == 1 and len(matmul.input) == 2 and matmul.input[1] == transpose.output[0], 'Original MatMul mismatch')
        add = one_consumer(matmul.output[0], 'Add')
        require(len(add.output) == 1 and set(add.input) == {matmul.output[0], b.name} and one_consumer(b.name, 'Add') is add,
                'Original FC1 bias Add mismatch')
        # Leave the exact original GELU nodes/operand order untouched.
        hidden = add.output[0]
        require(any(n.op_type == 'Div' and n.input[0] == hidden for n in consumers.get(hidden, []))
                and any(n.op_type == 'Mul' and hidden in n.input for n in consumers.get(hidden, [])), 'Original exact GELU boundary missing')
        branch_nodes, branch_weights, branch_biases, branch_outputs = [], [], [], []
        prefix = f'camera_fc1_split4_{index}/'
        for part in range(4):
            start, stop = part * 2048, (part + 1) * 2048
            wt, bt = np.ascontiguousarray(wa[start:stop].T), np.ascontiguousarray(ba[start:stop])
            wn, bn, mm, biased = [prefix + f'{name}_{part}' for name in ('weight', 'bias', 'matmul', 'biased')]
            require(not any(value in consumers for value in (wn, bn, mm, biased)), 'New names collide with parent graph')
            tensors.extend([nh.from_array(wt, wn), nh.from_array(bt, bn)])
            branch_nodes.append(h.make_node('MatMul', [matmul.input[0], wn], [mm], name=mm))
            add_inputs = [bn if x == b.name else mm for x in add.input]
            branch_nodes.append(h.make_node('Add', add_inputs, [biased], name=biased))
            branch_outputs.append(biased)
            branch_weights.append(wt.T); branch_biases.append(bt)
        branch_nodes.append(h.make_node('Concat', branch_outputs, [hidden], axis=-1, name=prefix + 'concat'))
        raw_w = hashlib.sha256(wa.tobytes()).hexdigest()
        raw_b = hashlib.sha256(ba.tobytes()).hexdigest()
        require(hashlib.sha256(np.concatenate(branch_weights).tobytes()).hexdigest() == raw_w
                and hashlib.sha256(np.concatenate(branch_biases).tobytes()).hexdigest() == raw_b, 'Original parameter reconstruction failed')
        removed_nodes.update(n.name for n in (transpose, matmul, add))
        removed_initializers.update((w.name, b.name))
        removed_values.update((transpose.output[0], matmul.output[0]))
        substitutions[matmul.name] = branch_nodes
        replacements.append(dict(index=index, original_weight=w.name, original_bias=b.name,
                                 weight_raw_sha256=raw_w, bias_raw_sha256=raw_b, chunks=4,
                                 hidden_output=hidden, original_fc1_nodes=[n.name for n in (transpose, matmul, add)],
                                 output_channel_ranges=[[p * 2048, (p + 1) * 2048] for p in range(4)]))
    require(len(replacements) == 4 and len(removed_nodes) == 12 and len(removed_initializers) == 8, 'Exactly four FC1 edits required')
    original_nodes = [n.SerializeToString() for n in nodes if n.name not in removed_nodes]
    original_tensors = [t.SerializeToString() for t in initializers if t.name not in removed_initializers]
    original_io = [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)]
    new_nodes = []
    for node in nodes:
        if node.name in substitutions:
            new_nodes.extend(substitutions[node.name])
        elif node.name not in removed_nodes:
            new_nodes.append(node)
    del model.graph.node[:]; model.graph.node.extend(new_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend([t for t in initializers if t.name not in removed_initializers] + tensors)
    retained_value_info = [v for v in model.graph.value_info if v.name not in removed_values]
    del model.graph.value_info[:]; model.graph.value_info.extend(retained_value_info)
    new_names = {n.name for branch in substitutions.values() for n in branch}
    require(original_nodes == [n.SerializeToString() for n in model.graph.node if n.name not in new_names], 'Other original nodes changed')
    require(original_tensors == [t.SerializeToString() for t in model.graph.initializer if t.name not in {x.name for x in tensors}], 'Other parameters changed')
    require(original_io == [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)], 'Original step ABI changed')
    require(sum(int(np.prod(t.dims)) for t in tensors) == 4 * (8192 * 2048 + 8192), 'FC1 weight duplication or omission')
    onnx.checker.check_model(model, full_check=True)
    onnx.save_model(model, str(destination))
    return dict(replacements=replacements, all_other_original_nodes_protobuf_unchanged=True,
                all_other_original_initializers_protobuf_unchanged=True, all_original_FC1_parameter_values_preserved=True,
                FC1_weights_not_duplicated=True, original_GELU_nodes_protobuf_unchanged=True,
                original_IO_protobuf_unchanged=True, original_nodes=len(nodes), candidate_nodes=len(new_nodes),
                original_initializers=len(initializers), candidate_initializers=len(model.graph.initializer), ONNX_version=onnx.__version__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', type=Path, default=Path('/mnt/vggt-data/vggt-slam2/experiments/vggt-v16-camera-step/export112-r1/camera-step.onnx'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--export-only', action='store_true')
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588 board required')
    helper_path = Path(__file__).with_name('compile_camera_fc1_splits_board.py')
    require(sha(helper_path) == HELPER_SHA and sha(a.onnx) == PARENT_SHA, 'Frozen helper or original step graph mismatch')
    spec = importlib.util.spec_from_file_location('_original_camera_compile', helper_path)
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    source, root = a.onnx.resolve(), a.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    r = dict(schema='v69_original_CameraHead_step_FC1_split_compile', status='running', chunks=4,
             script_sha256=sha(__file__), helper_sha256=HELPER_SHA, baseline_onnx_sha256=PARENT_SHA,
             trunk_blocks_per_step=4, required_outer_iterations=4, candidate_accepted=False,
             full_model_or_realtime_validated=False, model_inference_executed=False, configuration=c.CONFIG,
             build_options=c.BUILD, operations=[], released=None, cleanup_errors=[])
    def write():
        tmp = root / 'result.json.tmp'; tmp.write_text(json.dumps(r, indent=2, allow_nan=False) + '\n'); tmp.replace(root / 'result.json')
    cwd, temp, sdk, started = Path.cwd(), os.environ.get('TMPDIR'), None, time.monotonic()
    pins = {str(source): PARENT_SHA, str(helper_path.resolve()): HELPER_SHA, str(Path(__file__).resolve()): r['script_sha256']}
    write()
    try:
        r['graph_proof'] = rewrite(source, root / 'camera-step-split4.onnx')
        r['onnx'] = c.artifact(root / 'camera-step-split4.onnx')
        pins[str(root / r['onnx']['file'])] = r['onnx']['sha256']
        r['status'] = 'exported_not_runtime_validated'; write()
        if not a.export_only:
            r['RKNN_Toolkit2_version'] = c.importlib.metadata.version('rknn-toolkit2')
            require(r['RKNN_Toolkit2_version'].split('+')[0] == '2.3.2', 'RKNN2.3.2 required')
            os.chdir(root); os.environ['TMPDIR'] = str(root)
            from rknn.api import RKNN
            sdk = RKNN(verbose=True, verbose_file=str(root / 'sdk.log')); r['released'] = False
            for name, fn in [('config', lambda: sdk.config(**c.CONFIG)),
                             ('load_onnx', lambda: sdk.load_onnx(model=str(root / r['onnx']['file']))),
                             ('build', lambda: sdk.build(**c.BUILD)),
                             ('export_rknn', lambda: sdk.export_rknn(str(root / 'camera-step-split4.rknn')))]:
                row = dict(name=name, status='running'); r['operations'].append(row); write(); t = time.monotonic()
                code = fn(); row.update(status='completed', code=code, seconds=time.monotonic() - t)
                require(code in (None, 0), f'{name} returned {code}')
            r['rknn'] = c.artifact(root / 'camera-step-split4.rknn'); r['status'] = 'compiled_not_quality_accepted'
    except BaseException as exc:
        r.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if r['operations'] and r['operations'][-1]['status'] == 'running':
            r['operations'][-1].update(status='failed', error=r['error'])
    finally:
        if sdk is not None:
            try:
                r['release_code'] = sdk.release(); r['released'] = r['release_code'] in (None, 0)
                require(r['released'], f'release returned {r["release_code"]}')
            except BaseException as exc:
                r['cleanup_errors'].append(f'release: {type(exc).__name__}: {exc}')
        try:
            os.chdir(cwd)
            if temp is None: os.environ.pop('TMPDIR', None)
            else: os.environ['TMPDIR'] = temp
            r['environment_restored'] = Path.cwd() == cwd and os.environ.get('TMPDIR') == temp
            r['source_and_inputs_unchanged'] = all(sha(Path(path)) == value for path, value in pins.items())
            require(r['environment_restored'] and r['source_and_inputs_unchanged'], 'Source/environment changed')
        except BaseException as exc:
            r['cleanup_errors'].append(f'restore/identity: {type(exc).__name__}: {exc}')
        if r['cleanup_errors']: r['status'] = 'failed'
        r['seconds'] = time.monotonic() - started; write()
        print(json.dumps(dict(status=r['status'], result_sha256=sha(root / 'result.json'))), flush=True)
    return int(r['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
