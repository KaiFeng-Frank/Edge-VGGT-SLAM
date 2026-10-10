#!/usr/bin/env python3
"""Board-only ORT check: one-step checks and two independent four-step recurrences.

Closed-loop ORT always conditions on its own unactivated cumulative prediction.
Feeding captured CPU conditioning is a separate one-step diagnostic. All CPU
reference execution is on the board; no host or NPU inference is performed.
"""
from __future__ import annotations

import argparse
import gc
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
EXPORT_SHA = 'b26efe9c6cd62a2b7977e6e9c096cf0bd9bbf63076a063b9a332ef59565af258'
CHECK_V15_SHA = '3d7740cac2d94b17dafa2ed822944ad09062c3441c2a1acc214a649b268c0c0b'


def load(name, digest, module_name):
    import hashlib
    path = Path(__file__).with_name(name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Frozen helper SHA mismatch: {name}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def run(args, step, checker):
    report = dict(kind='shared_CameraHead_step_board_CPU_ORT_closed_loop_check', status='failed',
                  script_sha256=step.sha(__file__), CPU_original_head_calls=0, ORT_call_count=0,
                  original_iterations=4, host_model_inference_executed=False, npu_used=False, ORB_used=False,
                  production_modified=False, full_model_quality_validated=False, real_time_validated=False,
                  tolerance=dict(rtol=checker.RTOL, atol=checker.ATOL, predefined=True),
                  artifacts={}, runs=[], cleanup_errors=[])
    pins, hooks, allowed, start = {}, [], None, time.monotonic()
    head = state = session = None
    try:
        old, old_path, capture, capture_path = step.load_frozen()
        report['device_tree_compatible'] = capture.require_board()
        helper, helper_path = capture.base_capture()
        pins = {str(args.model): step.require_sha(args.model, args.model_sha256),
                str(args.export_receipt): step.require_sha(args.export_receipt, args.export_sha256),
                str(args.capture_dir / 'capture.json'): step.require_sha(args.capture_dir / 'capture.json', args.capture_sha256),
                str(Path(__file__).resolve()): step.sha(__file__)}
        for name, digest in [('export-v16-camera-step-board.py', EXPORT_SHA),
                             ('check-v15-camera-onnx-board.py', CHECK_V15_SHA)]:
            path = Path(__file__).with_name(name)
            pins[str(path)] = step.require_sha(path, digest)
        pins.update({str(old_path): step.EXPORT_V15_SHA, str(capture_path): step.CAPTURE_SHA,
                     str(helper_path): capture.BASE_CAPTURE_SHA})
        metadata = json.loads((args.capture_dir / 'capture.json').read_text())
        exported = json.loads(args.export_receipt.read_text())
        if (metadata.get('status') != 'completed' or metadata.get('script_sha256') != step.CAPTURE_SHA
                or metadata.get('npu_used') is not False or metadata.get('source_and_inputs_unchanged') is not True
                or metadata.get('cleanup_errors') or exported.get('script_sha256') != EXPORT_SHA
                or exported.get('status') != 'exported_and_structurally_checked' or exported.get('cleanup_errors')
                or exported.get('capture', {}).get('sha256') != args.capture_sha256
                or exported.get('architecture') != metadata.get('architecture')
                or exported.get('onnx', {}).get('sha256') != args.model_sha256
                or exported.get('weight_sharing', {}).get('original_parameters_verified') != 68
                or exported.get('source_and_inputs_unchanged') is not True):
            raise ValueError('Exact original capture/shared-step export provenance required')
        originals = {**helper.SOURCE_PINS, **capture.EXTRA_PINS}
        if metadata.get('source_hashes') != originals:
            raise ValueError('Original source provenance mismatch')
        for name, digest in originals.items():
            pins[str(SOURCE / name)] = step.require_sha(SOURCE / name, digest)
        for name, item in metadata['artifacts'].items():
            path = args.capture_dir / name
            pins[str(path)] = step.require_sha(path, item['sha256'])
            if path.stat().st_size != item['bytes']:
                raise ValueError('Capture artifact bytes mismatch')
        allowed = set(os.sched_getaffinity(0))
        if not capture.CPUS <= allowed:
            raise ValueError('CPU 4,5,6,7 must be allowed')
        os.sched_setaffinity(0, capture.CPUS)
        os.environ.update({key: '4' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/vggt')]
        import numpy as np
        import torch
        from vggt.heads.camera_head import CameraHead
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if torch.__version__ != metadata['runtime']['torch']:
            raise ValueError('Exact captured board Torch required')
        tokens = np.load(args.capture_dir / 'input.npy', allow_pickle=False)
        if helper.array_info(tokens)['sha256'] != metadata['artifacts']['input.npy']['tensor_sha256'] or tokens.shape != step.INPUT_SHAPE:
            raise ValueError('Original raw tokens differ')
        with np.load(args.capture_dir / 'pose-iterations.npz', allow_pickle=False) as saved:
            archived = {key: saved[key].copy() for key in saved.files}
        with np.load(args.capture_dir / 'camera-state.npz', allow_pickle=False) as saved:
            state = {key: np.ascontiguousarray(saved[key]) for key in saved.files}
        weights = metadata['artifacts']['camera-state.npz']['tensors']
        if len(state) != 69 or set(state) != set(weights) or any(helper.array_info(value) != weights[key] for key,value in state.items()):
            raise ValueError('All 69 original parameter values must match')
        seed = np.broadcast_to(state['empty_pose_tokens'], step.POSE_SHAPE).copy()
        with torch.device('meta'):
            head = CameraHead()
        head.load_state_dict({key:torch.from_numpy(value) for key,value in state.items()}, strict=True, assign=True)
        head.eval().requires_grad_(False)
        if capture.camera_contract(head) != metadata['architecture']:
            raise ValueError('Original four-iteration architecture mismatch')
        std = float(np.std(tokens.astype(np.float64)))
        if not 0 < std < float('inf'):
            raise ValueError('Finite nonconstant original camera tokens required')
        second = np.ascontiguousarray(tokens + (0.05 * std * np.sin(np.arange(tokens.size) * 0.6180339887498949)).astype(np.float32).reshape(tokens.shape))
        report['second_input_definition'] = dict(formula='x + float32(.05 * std64(x) * sin(arange * .6180339887498949))',
                                                 parameters_chosen_before_results=True, std64=std)
        report['artifacts']['input-01.npy'] = checker.save(args.output_dir, 'input-01.npy', second, helper)
        conditions, deltas = [], []
        hooks = [head.embed_pose.register_forward_pre_hook(lambda module,args:conditions.append(args[0].detach().numpy().copy())),
                 head.pose_branch.register_forward_hook(lambda module,args,result:deltas.append(result.detach().numpy().copy()))]
        references = []
        with torch.inference_mode():
            for index, value in enumerate((tokens, second)):
                conditions.clear();deltas.clear()
                poses = head([torch.from_numpy(value).unsqueeze(2)], num_iterations=4)
                report['CPU_original_head_calls'] += 1
                refs = dict(conditions=[x.copy() for x in conditions], deltas=[x.copy() for x in deltas],
                            poses=[x.detach().numpy().copy() for x in poses])
                if len(refs['conditions']) != 4 or len(refs['deltas']) != 4 or len(refs['poses']) != 4:
                    raise ValueError('Original head must produce all four iterations')
                if index == 0 and any(not np.array_equal(array, archived[f'iteration_{i}']) for i,array in enumerate(refs['poses'])):
                    raise ValueError('Original head differs from the captured four-iteration reference')
                references.append(refs)
        for hook in hooks:
            hook.remove()
        hooks.clear()
        del head, state, poses
        head = state = None
        gc.collect()
        sys.path.insert(0, str(args.ort_site))
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
        session.disable_fallback()
        if session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Explicit CPU-only ORT validation required')
        ins, outs = session.get_inputs(), session.get_outputs()
        if ({value.name:(value.shape,value.type) for value in ins} != {
                'camera_tokens':(list(step.INPUT_SHAPE),'tensor(float)'),
                'conditioning_pose':(list(step.POSE_SHAPE),'tensor(float)')}
                or [(v.name,v.shape,v.type) for v in outs] != [('pose_delta',list(step.POSE_SHAPE),'tensor(float)')]):
            raise ValueError('Exact two-input/single-delta ONNX bindings required')
        gate_checks, ort_final = [], []
        for index,(raw,ref) in enumerate(zip((tokens,second),references)):
            evidence, arrays = dict(input_index=index, independent_one_step=[], closed_loop=[]), {}
            for i in range(4):
                delta = session.run(['pose_delta'],dict(camera_tokens=raw,conditioning_pose=ref['conditions'][i]))[0]
                report['ORT_call_count'] += 1
                comparison = checker.compare(delta, ref['deltas'][i], np)
                evidence['independent_one_step'].append(dict(iteration=i, comparison=comparison))
                gate_checks.append(comparison['allclose'])
                arrays[f'one_step_delta_{i}'] = delta.copy()
            prediction = None
            for i in range(4):
                conditioning = seed.copy() if prediction is None else prediction.copy()
                delta = session.run(['pose_delta'],dict(camera_tokens=raw,conditioning_pose=conditioning))[0]
                report['ORT_call_count'] += 1
                prediction = step.accumulate_delta(prediction, delta)
                activated = step.activate_numpy_pose(prediction)
                dc, pc = checker.compare(delta, ref['deltas'][i], np), checker.compare(activated, ref['poses'][i], np)
                gate_checks.extend([dc['allclose'],pc['allclose']])
                evidence['closed_loop'].append(dict(iteration=i, own_unactivated_prediction_used=True,
                                                    delta_vs_original_cpu=dc, activated_vs_original_cpu=pc))
                arrays.update({f'closed_conditioning_{i}':conditioning, f'closed_delta_{i}':delta.copy(),
                               f'closed_preactivation_{i}':prediction.copy(),f'closed_activation_{i}':activated})
            path = args.output_dir / f'iterations-{index:02d}.npz'
            with path.open('xb') as stream:
                np.savez(stream, **arrays)
            report['artifacts'][path.name] = dict(sha256=step.sha(path), bytes=path.stat().st_size,
                                                  tensors={k:helper.array_info(v) for k,v in arrays.items()})
            report['runs'].append(evidence)
            ort_final.append(activated.copy())
        report['CPU_second_output_changed'] = not np.array_equal(references[0]['poses'][-1],references[1]['poses'][-1])
        report['ORT_second_output_changed'] = not np.array_equal(*ort_final)
        report['source_and_input_hashes'] = pins
        report['source_and_input_hashes_after'] = {path:step.sha(path) for path in pins}
        report['source_and_inputs_unchanged'] = pins == report['source_and_input_hashes_after']
        if (not all(gate_checks) or not report['CPU_second_output_changed'] or not report['ORT_second_output_changed']
                or not report['source_and_inputs_unchanged']):
            raise ValueError('Shared-step ONNX failed predefined numerics/input-dependence/source gates')
        report['runtime'] = dict(torch=torch.__version__, onnxruntime=ort.__version__, threads=4,
                                 cpu_affinity=sorted(os.sched_getaffinity(0)))
        report['closed_loop_recurrence_validated'] = True
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except BaseException as exc:
                report['cleanup_errors'].append(f'Hook cleanup: {exc}')
        head = state = session = None
        gc.collect()
        if allowed is not None:
            try:
                os.sched_setaffinity(0, allowed)
                report['affinity_restored'] = set(os.sched_getaffinity(0)) == allowed
            except BaseException as exc:
                report['cleanup_errors'].append(f'Affinity cleanup: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['total_seconds'] = time.monotonic() - start
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        path = args.output_dir / 'check.json'
        with path.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','export-receipt','capture-dir','output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('model-sha256','export-sha256','capture-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--ort-site', type=Path, default=BASE / 'experiments/orb-v12-dino-block/ort-site')
    args = parser.parse_args()
    step,_ = load('export-v16-camera-step-board.py', EXPORT_SHA, '_v16_step_contract')
    checker,_ = load('check-v15-camera-onnx-board.py', CHECK_V15_SHA, '_v16_frozen_numeric_contract')
    _,_,capture,_ = step.load_frozen()
    capture.require_board()
    if Path.cwd().resolve() != BASE or Path(__file__).resolve().parents[3] != SOURCE:
        parser.error('Run deployed source from board BASE')
    if not args.output_dir.resolve().is_relative_to(step.EXPERIMENT) or os.path.lexists(args.output_dir):
        parser.error('Use a NEW output directory below experiments/vggt-v16-camera-step')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = run(args, step, checker)
    path = args.output_dir / 'check.json'
    print(json.dumps(dict(status=result['status'], error=result.get('error'), output=str(path), sha256=step.sha(path))))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
