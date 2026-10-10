#!/usr/bin/env python3
"""Export one original CameraHead refinement step, on RK3588S only.

Raw camera tokens + an unactivated conditioning pose -> delta. Four calls to
the same graph retain the original four iterations. The learned empty pose is
saved separately; the graph contains the other 68 original parameter tensors.
No host model inference, production change or whole-model quality claim.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
EXPERIMENT = BASE / 'experiments/vggt-v16-camera-step'
INPUT_SHAPE, POSE_SHAPE = (1, 2, 2048), (1, 2, 9)
EXPORT_V15_SHA = '3f97e96e6e9f7beaf5a5de56569774aa79fa17eb7e30664e54cdbbfec224a033'
CAPTURE_SHA = 'd50c217354b2f48f50626ff1ea2c98671d5611743ae0e97bed1c04b2991341a0'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected) or sha(path) != expected):
        raise ValueError(f'Explicit SHA256 mismatch: {path}')
    return expected


def load_frozen():
    path = Path(__file__).with_name('export-v15-camera-board.py')
    require_sha(path, EXPORT_V15_SHA)
    spec = importlib.util.spec_from_file_location('_v16_original_camera_export_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    capture, capture_path = module.load_capture_helper()
    return module, path, capture, capture_path


def pose_array(value):
    import numpy as np
    array = np.asarray(value)
    if (array.shape != POSE_SHAPE or array.dtype != np.float32
            or not array.flags.c_contiguous or not np.isfinite(array).all()):
        raise ValueError('Finite contiguous FP32 [1,2,9] pose required')
    return array


def accumulate_delta(previous, delta):
    """Original first assignment, then FP32 addition; never feed activation back."""
    delta = pose_array(delta)
    return delta.copy() if previous is None else pose_array(previous) + delta


def activate_numpy_pose(prediction):
    """Exact original linear translation/quaternion and ReLU FoV, 18 scalars."""
    result = pose_array(prediction).copy()
    fov = result[..., 7:]
    fov[fov < 0] = 0
    return result


def make_step(torch, head):
    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.camera_head = head

        def forward(self, camera_tokens, conditioning_pose):
            h = self.camera_head
            tokens = h.token_norm(camera_tokens)
            module_input = h.embed_pose(conditioning_pose)
            shift, scale, gate = h.poseLN_modulation(module_input).chunk(3, dim=-1)
            modulated = gate * (h.adaln_norm(tokens) * (1 + scale) + shift) + tokens
            return h.pose_branch(h.trunk_norm(h.trunk(modulated)))
    return Step().eval()


def both_inputs_reach_output(model):
    dependencies = {node_output: list(node.input) for node in model.graph.node for node_output in node.output}
    reached, todo = set(), [model.graph.output[0].name]
    while todo:
        name = todo.pop()
        if name not in reached:
            reached.add(name)
            todo.extend(dependencies.get(name, ()))
    return all(value.name in reached for value in model.graph.input)


def export(output, capture_dir, capture_sha, onnx_site):
    report = dict(kind='original_shared_CameraHead_single_step_board_ONNX_export', status='failed',
                  script_sha256=sha(__file__), camera_iterations=4, trunk_blocks_per_step=4,
                  host_model_inference_executed=False, npu_used=False, ORB_used=False,
                  production_modified=False, real_time_validated=False, full_model_quality_validated=False,
                  original_head_reference_calls=0, CPU_step_reference_calls=0,
                  board_export_trace_step_calls=0, cleanup_errors=[], artifacts={})
    pins, hooks, allowed, started = {}, [], None, time.monotonic()
    try:
        old, old_path, capture, capture_path = load_frozen()
        report['device_tree_compatible'] = capture.require_board()
        helper, helper_path = capture.base_capture()
        receipt_path = capture_dir / 'capture.json'
        pins[str(receipt_path)] = require_sha(receipt_path, capture_sha)
        meta = json.loads(receipt_path.read_text())
        if (meta.get('status') != 'completed' or meta.get('script_sha256') != CAPTURE_SHA
                or meta.get('kind') != 'original_complete_CameraHead_CPU_capture'
                or meta.get('camera_iterations') != 4 or meta.get('CPU_camera_forward_call_count') != 3
                or meta.get('CPU_trunk_call_count') != 48 or meta.get('npu_used') is not False
                or meta.get('ORB_used') is not False or meta.get('cleanup_errors')
                or meta.get('source_and_inputs_unchanged') is not True
                or meta.get('checkpoint_stat_unchanged') is not True
                or meta.get('checkpoint', {}).get('sha256') != capture.CHECKPOINT_SHA
                or meta.get('cpu_reference', {}).get('all_elementwise_equal') is not True):
            raise ValueError('Frozen original four-iteration CPU capture is required')
        originals = {**helper.SOURCE_PINS, **capture.EXTRA_PINS}
        if meta.get('source_hashes') != originals:
            raise ValueError('Capture original source identity mismatch')
        pins.update({str(old_path): EXPORT_V15_SHA, str(capture_path): CAPTURE_SHA,
                     str(helper_path): capture.BASE_CAPTURE_SHA, str(Path(__file__).resolve()): sha(__file__)})
        for name, digest in originals.items():
            pins[str(SOURCE / name)] = require_sha(SOURCE / name, digest)
        for item in meta['input_images']:
            pins[item['path']] = require_sha(item['path'], item['sha256'])
        allowed = set(os.sched_getaffinity(0))
        if not capture.CPUS <= allowed:
            raise ValueError('CPU 4,5,6,7 must be allowed for exact reference/export tracing')
        os.sched_setaffinity(0, capture.CPUS)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '4'
        sys.path[:0] = [str(onnx_site), str(SOURCE), str(SOURCE / 'third_party/vggt')]
        import numpy as np
        import torch
        import onnx
        from onnx import numpy_helper
        from vggt.heads.camera_head import CameraHead
        from vggt.heads.head_act import activate_pose
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if onnx.__version__ != '1.16.2' or torch.__version__ != meta['runtime']['torch']:
            raise ValueError('Use the captured board Torch and isolated ONNX 1.16.2')
        report['runtime'] = dict(torch=torch.__version__, onnx=onnx.__version__, numpy=np.__version__,
                                 cpu_affinity=sorted(os.sched_getaffinity(0)), threads=4)
        for name in ('input.npy', 'output.npy', 'camera-state.npz', 'pose-iterations.npz'):
            path, item = capture_dir / name, meta['artifacts'][name]
            pins[str(path)] = require_sha(path, item['sha256'])
            if path.stat().st_size != item['bytes']:
                raise ValueError('Capture artifact bytes mismatch')
        raw = np.load(capture_dir / 'input.npy', allow_pickle=False)
        final = np.load(capture_dir / 'output.npy', allow_pickle=False)
        for name, array, shape in [('input.npy', raw, INPUT_SHAPE), ('output.npy', final, POSE_SHAPE)]:
            info = helper.array_info(array)
            if (tuple(array.shape) != shape or not array.flags.c_contiguous
                    or meta['artifacts'][name]['tensor_sha256'] != info['sha256']):
                raise ValueError('Exact raw token/reference identity mismatch')
        with np.load(capture_dir / 'pose-iterations.npz', allow_pickle=False) as saved:
            poses = {key: saved[key].copy() for key in saved.files}
        if set(poses) != {f'iteration_{i}' for i in range(4)}:
            raise ValueError('Exactly four captured original activated poses required')
        for key, array in poses.items():
            if helper.array_info(array) != meta['artifacts']['pose-iterations.npz']['tensors'][key]:
                raise ValueError('Original activated pose SHA mismatch')
        if not np.array_equal(final, poses['iteration_3']):
            raise ValueError('Original final output is not iteration four')
        with np.load(capture_dir / 'camera-state.npz', allow_pickle=False) as saved:
            state = {key: np.ascontiguousarray(saved[key]) for key in saved.files}
        weights = meta['artifacts']['camera-state.npz']['tensors']
        if len(state) != 69 or set(state) != set(weights):
            raise ValueError('Exact 69 original parameter tensors required')
        for key, array in state.items():
            if helper.array_info(array) != weights[key]:
                raise ValueError(f'Original weight mismatch: {key}')
        with torch.device('meta'):
            head = CameraHead()
        head.load_state_dict({key: torch.from_numpy(array) for key, array in state.items()}, strict=True, assign=True)
        head.eval().requires_grad_(False)
        if capture.camera_contract(head) != meta['architecture']:
            raise ValueError('Original head architecture/epsilon mismatch')
        raw_tensor = torch.from_numpy(raw)
        original_conditions, original_deltas = [], []
        hooks = [head.embed_pose.register_forward_pre_hook(
                     lambda module, args: original_conditions.append(args[0].detach().numpy().copy())),
                 head.pose_branch.register_forward_hook(
                     lambda module, args, result: original_deltas.append(result.detach().numpy().copy()))]
        with torch.no_grad():
            original = head([raw_tensor.unsqueeze(2)], num_iterations=4)
        report['original_head_reference_calls'] = 1
        for hook in hooks:
            hook.remove()
        hooks.clear()
        if (len(original_conditions) != 4 or len(original_deltas) != 4
                or any(not np.array_equal(value.numpy(), poses[f'iteration_{i}']) for i, value in enumerate(original))):
            raise ValueError('Current original head does not reproduce all four frozen CPU poses')
        step = make_step(torch, head)
        seed = np.broadcast_to(state['empty_pose_tokens'], POSE_SHAPE).copy()
        prediction, iteration_arrays, checks = None, {}, []
        for i in range(4):
            conditioning = seed if prediction is None else prediction.copy()
            with torch.no_grad():
                delta = step(raw_tensor, torch.from_numpy(conditioning)).numpy().copy()
            prediction = accumulate_delta(prediction, delta)
            activated = activate_pose(torch.from_numpy(prediction), trans_act=head.trans_act,
                                      quat_act=head.quat_act, fl_act=head.fl_act).numpy().copy()
            row = dict(iteration=i, conditioning_array_equal=np.array_equal(conditioning, original_conditions[i]),
                       delta_array_equal=np.array_equal(delta, original_deltas[i]),
                       activated_array_equal=np.array_equal(activated, poses[f'iteration_{i}']),
                       explicit_FP32_activation_array_equal=np.array_equal(activate_numpy_pose(prediction), activated))
            if not all(value for key, value in row.items() if key != 'iteration'):
                raise ValueError(f'Single-step decomposition changed original iteration {i}')
            iteration_arrays.update({f'conditioning_{i}': conditioning, f'delta_{i}': delta,
                                     f'preactivation_{i}': prediction.copy(), f'activation_{i}': activated})
            checks.append(row)
        report['CPU_step_reference_calls'] = 4
        report['reference_iterations'] = checks
        for name, array in [('conditioning-seed.npy', seed), ('empty-pose-token.npy', state['empty_pose_tokens'])]:
            report['artifacts'][name] = capture.save_array(output, name, array, helper)
        steps_path = output / 'step-iterations.npz'
        with steps_path.open('xb') as stream:
            np.savez(stream, **iteration_arrays)
        report['artifacts'][steps_path.name] = dict(sha256=sha(steps_path), bytes=steps_path.stat().st_size,
                tensors={key: helper.array_info(array) for key, array in iteration_arrays.items()})
        traces = []
        hooks = [step.register_forward_hook(lambda module, args, result: traces.append(result.detach().numpy().copy()))]
        model_path, tick = output / 'camera-step.onnx', time.perf_counter()
        with torch.no_grad():
            torch.onnx.export(step, (raw_tensor, torch.from_numpy(seed)), str(model_path),
                              input_names=['camera_tokens', 'conditioning_pose'], output_names=['pose_delta'],
                              export_params=True, opset_version=17, do_constant_folding=False,
                              keep_initializers_as_inputs=False, training=torch.onnx.TrainingMode.EVAL)
        report['export_seconds_including_board_trace_and_IO'] = time.perf_counter() - tick
        for hook in hooks:
            hook.remove()
        hooks.clear()
        report['board_export_trace_step_calls'] = len(traces)
        if not traces or any(not np.array_equal(value, original_deltas[0]) for value in traces):
            raise ValueError('Single-step export trace changed first original delta')
        if not 0 < model_path.stat().st_size <= 1024 ** 3:
            raise ValueError('Single-step ONNX exceeds the 1GiB bounded budget')
        del step, head, state, original
        gc.collect()
        model = onnx.load(str(model_path), load_external_data=False)
        bindings = [('camera_tokens', INPUT_SHAPE), ('conditioning_pose', POSE_SHAPE)]
        def shape(value):
            return tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        if ([(v.name, shape(v)) for v in model.graph.input] != bindings
                or [(v.name, shape(v)) for v in model.graph.output] != [('pose_delta', POSE_SHAPE)]
                or any(v.type.tensor_type.elem_type != 1 for v in (*model.graph.input, *model.graph.output))
                or not both_inputs_reach_output(model)):
            raise ValueError('Both raw tokens and dynamic conditioning must reach the single FP32 delta output')
        step_weights = {key: value for key, value in weights.items() if key != 'empty_pose_tokens'}
        report['weight_sharing'] = old.weight_bindings(model, step_weights, np, numpy_helper)
        if any(tensor.name == 'camera_head.empty_pose_tokens' for tensor in model.graph.initializer):
            raise ValueError('Learned empty pose must be external seed, not a step initializer')
        onnx.checker.check_model(str(model_path), full_check=True)
        report['onnx'] = dict(path=str(model_path), bytes=model_path.stat().st_size, sha256=sha(model_path),
                            input_bindings=[dict(index=i, name=name, shape=list(size)) for i,(name,size) in enumerate(bindings)],
                            output_bindings=[dict(index=0, name='pose_delta', shape=list(POSE_SHAPE))],
                            node_count=len(model.graph.node), initializer_count=len(model.graph.initializer),
                            both_dynamic_inputs_reach_output=True, opset=17, constant_folding=False)
        report['capture'] = dict(path=str(receipt_path), sha256=capture_sha, checkpoint=meta['checkpoint'],
                                 artifacts={name: meta['artifacts'][name] for name in ('input.npy','output.npy','pose-iterations.npz')})
        report['architecture'] = meta['architecture']
        report['outer_iteration_contract'] = dict(iterations=4, first_conditioning='original_empty_pose_tokens_expanded',
                later_conditioning='previous_unactivated_FP32_prediction', first_update='prediction = delta',
                later_update='prediction = prediction + delta', final_activation='linear/linear/ReLU',
                FP32_scalar_additions_after_first_step=54, activation_scalars=18,
                all_four_original_iterations_array_equal=True, empty_weight=weights['empty_pose_tokens'])
        report['source_and_input_hashes'] = pins
        report['source_and_input_hashes_after'] = {path: sha(path) for path in pins}
        report['source_and_inputs_unchanged'] = pins == report['source_and_input_hashes_after']
        if not report['source_and_inputs_unchanged']:
            raise ValueError('Original source/state/input changed during export')
        report['status'] = 'exported_and_structurally_checked'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except BaseException as exc:
                report['cleanup_errors'].append(f'Hook cleanup: {exc}')
        if allowed is not None:
            try:
                os.sched_setaffinity(0, allowed)
                report['affinity_restored'] = set(os.sched_getaffinity(0)) == allowed
            except BaseException as exc:
                report['cleanup_errors'].append(f'Affinity cleanup: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report['total_seconds_including_hash_reference_trace_IO'] = time.monotonic() - started
        with (output / 'export.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, default=BASE / 'experiments/vggt-v15-camera/capture112-r1')
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--onnx-site', type=Path, default=BASE / 'experiments/vggt-v15-camera/onnx-site')
    args = parser.parse_args()
    _, _, capture, _ = load_frozen()
    capture.require_board()
    if Path.cwd().resolve() != BASE or Path(__file__).resolve().parents[3] != SOURCE:
        parser.error('Run deployed source from board /mnt/vggt-data/vggt-slam2')
    output = args.output.resolve()
    if not output.is_relative_to(EXPERIMENT) or os.path.lexists(args.output) or output == args.capture_dir.resolve():
        parser.error('Use a NEW output directory below experiments/vggt-v16-camera-step')
    output.mkdir(parents=True, exist_ok=False)
    result = export(output, args.capture_dir.resolve(strict=True), args.capture_sha256, args.onnx_site.resolve(strict=True))
    path = output / 'export.json'
    print(json.dumps(dict(status=result['status'], error=result.get('error'), output=str(path), sha256=sha(path))))
    return 0 if result['status'] == 'exported_and_structurally_checked' else 1


if __name__ == '__main__':
    sys.exit(main())
