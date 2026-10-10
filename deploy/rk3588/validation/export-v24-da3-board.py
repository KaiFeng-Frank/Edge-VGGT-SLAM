#!/usr/bin/env python3
"""Board-only full DA3-SMALL depth/raw-pose export, then separate CPU ORT gate.

Official unconditioned weights/core are preserved. External normalized input is
S3HW, internally wrapped to B1S3HW. The learned depth/camera paths are exported;
official quaternion/FOV codec remains separately timed CPU arithmetic. DualDPT
ray outputs are discarded by the original default camera path, so unused ray
operations may be pruned by ONNX. This is not an NPU execution/performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import numpy as np

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SHAPE = (2, 3, 252, 336)
RTOL, ATOL = 1e-4, 1e-5
OUTPUTS = [dict(name='depth', reference_key='depth', shape=[2, 252, 336]),
           dict(name='depth_conf', reference_key='conf', shape=[2, 252, 336]),
           dict(name='pose_enc', reference_key='pose', shape=[2, 9])]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError(f'SHA256 mismatch: {Path(path).name}')


def record(value):
    value = np.asarray(value)
    if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError('Finite nonempty FP32 arrays required')
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype='float32', bytes=value.nbytes,
                sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def compare(actual, expected):
    record(actual)
    record(expected)
    if actual.shape != expected.shape:
        raise ValueError('Reference/output shapes differ')
    delta = actual.astype(np.float64)-expected.astype(np.float64)
    denominator = np.linalg.norm(expected.astype(np.float64).ravel())
    close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
    return dict(max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta**2))),
                relative_l2=float(np.linalg.norm(delta.ravel())/denominator) if denominator else None,
                array_equal=bool(np.array_equal(actual, expected)), finite=True,
                element_count=int(close.size), failed_elements=int(np.count_nonzero(~close)),
                elementwise_gate_passed=bool(close.all()), rtol=RTOL, atol=ATOL)


def require_board():
    if platform.machine() not in ('aarch64', 'arm64'):
        raise RuntimeError('Board-only helper; no host model execution permitted')
    path = Path('/proc/device-tree/compatible')
    compatible = path.read_bytes()
    if b'rockchip,rk3588' not in compatible:
        raise RuntimeError('RK3588/RK3588S device-tree identity required')
    return compatible.replace(b'\0', b' ').decode(errors='replace').strip()


def load_common(args):
    require_sha(args.common, args.common_sha256)
    spec = importlib.util.spec_from_file_location('v24_original_da3_common', args.common)
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    return common


def save_npy(output, filename, value):
    path = output/filename
    with path.open('xb') as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return dict(file=filename, bytes=path.stat().st_size, sha256=sha(path), array=record(value))


def weight_manifest(model):
    return {name: record(tensor.detach().cpu().contiguous().numpy())
            for name, tensor in model.state_dict().items()}


def bind_source_pins(args, common, pins):
    provenance = common.verify_source_manifest(args.source_dir, args.manifest, args.manifest_sha256)
    for relative, digest in provenance['source_hashes'].items():
        pins[str((args.source_dir/relative).resolve())] = digest
    config = args.source_dir.parent/'model-metadata'/'config.json'
    require_sha(config, common.MODEL_CONFIG_SHA256)
    pins[str(config.resolve())] = common.MODEL_CONFIG_SHA256
    return provenance


def make_wrapper(model):
    import torch

    class FullDepthPose(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = model

        def forward(self, images):
            feats, _ = self.net.backbone(images.unsqueeze(0), cam_token=None,
                                        export_feat_layers=[], ref_view_strategy='saddle_balanced')
            output = self.net.head(feats, 252, 336, patch_start_idx=0)
            pose = self.net.cam_dec(feats[-1][1])
            return output.depth.squeeze(0), output.depth_conf.squeeze(0), pose.squeeze(0)

    return FullDepthPose().eval()


def official_codec(pose):
    import torch
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse
    encoded = torch.from_numpy(np.ascontiguousarray(pose)).unsqueeze(0)
    start = time.perf_counter()
    c2w, intrinsics = pose_encoding_to_extri_intri(encoded, (252, 336))
    extrinsics = affine_inverse(c2w)
    seconds = time.perf_counter()-start
    return extrinsics.squeeze(0).numpy().copy(), intrinsics.squeeze(0).numpy().copy(), seconds


def export_model(args, report, pins):
    import torch
    import onnx
    if not torch.__version__.startswith('2.3.1'):
        raise ValueError('This first export must use the actual board Torch 2.3.1 CPU')
    if len(args.rgb_inputs or []) != 3 or len(args.rgb_sha256 or []) != 3:
        raise ValueError('Three original RGB input arrays and SHA pins required')
    common = load_common(args)
    require_sha(args.manifest, args.manifest_sha256)
    require_sha(args.checkpoint, args.checkpoint_sha256)
    pins.update({str(args.common): args.common_sha256, str(args.manifest): args.manifest_sha256,
                 str(args.checkpoint): args.checkpoint_sha256})
    bind_source_pins(args, common, pins)
    model, provenance = common.load_official_model(args.source_dir, args.manifest, args.manifest_sha256,
                                                   args.checkpoint, args.checkpoint_sha256)
    if model.training or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Official complete CPU FP32 eval model required')
    if (model.backbone.pretrained.embed_dim != 384 or len(model.backbone.pretrained.blocks) != 12
            or model.backbone.pretrained.num_heads != 6 or model.head.__class__.__name__ != 'DualDPT'
            or model.cam_dec is None):
        raise ValueError('The fixed official DA3-SMALL architecture is required')
    before = weight_manifest(model)
    released = {name:dict(shape=value['shape'], dtype=value['dtype'], bytes=value['bytes'],
                         tensor_sha256=value['sha256']) for name,value in before.items()}
    if released != provenance['parameters']:
        raise ValueError('Loaded complete model state differs from released weight metadata')
    report.update(official_model_provenance=provenance, original_state_metadata=before,
                  original_state_keys=len(before), original_parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=torch.__version__, onnx_version=onnx.__version__, artifacts={},
                  pairs=[], official_codec='official XYZW c2w quaternion/FOV -> affine_inverse w2c + K')
    inputs = []
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        for i, (path, digest) in enumerate(zip(args.rgb_inputs, args.rgb_sha256, strict=True)):
            require_sha(path, digest)
            pins[str(path)] = digest
            rgb = np.load(path, allow_pickle=False)
            if rgb.shape not in (SHAPE, (1,)+SHAPE) or rgb.dtype != np.float32:
                raise ValueError('Original RGB must be FP32 S3HW or B1S3HW with 2x252x336 pixels')
            record(rgb)
            if rgb.min() < 0 or rgb.max() > 1:
                raise ValueError('RGB source input must be [0,1], before one official ImageNet normalization')
            tick = time.perf_counter()
            images = common.normalize_rgb_input(rgb)
            preprocess_seconds = time.perf_counter()-tick
            if tuple(images.shape) != (1,)+SHAPE or images.dtype != torch.float32 or images.device.type != 'cpu':
                raise ValueError('Official normalizer output contract differs')
            images4 = images.squeeze(0).contiguous()
            inputs.append(images4)
            raw_pose = []
            hook = model.cam_dec.register_forward_hook(lambda _m, _a, value: raw_pose.append(value))
            try:
                tick = time.perf_counter()
                original = model(images, extrinsics=None, intrinsics=None, export_feat_layers=[],
                                 infer_gs=False, use_ray_pose=False, ref_view_strategy='saddle_balanced')
                reference_seconds = time.perf_counter()-tick
            finally:
                hook.remove()
            if len(raw_pose) != 1:
                raise ValueError('Exactly one original camera decoder invocation is required')
            values = dict(depth=original.depth.squeeze(0).numpy().copy(),
                          conf=original.depth_conf.squeeze(0).numpy().copy(),
                          pose=raw_pose[0].squeeze(0).numpy().copy(),
                          extrinsic=original.extrinsics.squeeze(0).numpy().copy(),
                          intrinsic=original.intrinsics.squeeze(0).numpy().copy())
            for item in OUTPUTS:
                if list(values[item['reference_key']].shape) != item['shape']:
                    raise ValueError('Original output shape differs from fixed schema')
            for value in values.values():
                record(value)
            E, K, codec_seconds = official_codec(values['pose'])
            codec = dict(extrinsic=compare(E, values['extrinsic']), intrinsic=compare(K, values['intrinsic']))
            if not all(v['elementwise_gate_passed'] for v in codec.values()):
                raise ValueError('Independent official CPU codec differs from original Net output')
            input_name, ref_name = f'normalized-input-pair-{i}.npy', f'reference-pair-{i}.npz'
            report['artifacts'][input_name] = save_npy(args.output, input_name, images4.numpy())
            with (args.output/ref_name).open('xb') as stream:
                np.savez(stream, **values)
            report['artifacts'][ref_name] = dict(file=ref_name, bytes=(args.output/ref_name).stat().st_size,
                                                sha256=sha(args.output/ref_name),
                                                arrays={k:record(v) for k,v in values.items()})
            report['pairs'].append(dict(index=i, rgb_file_sha256=digest, input_artifact=input_name,
                reference_artifact=ref_name, preprocessing_seconds=preprocess_seconds,
                official_reference_seconds=reference_seconds, official_codec_seconds=codec_seconds,
                codec_comparisons=codec, reference_hook='camera decoder output value reference; no copy/hash inside forward'))
        wrapper = make_wrapper(model)
        tick = time.perf_counter()
        actual = wrapper(inputs[0])
        wrapper_seconds = time.perf_counter()-tick
        with np.load(args.output/'reference-pair-0.npz', allow_pickle=False) as reference:
            checks = {item['name']: compare(value.numpy(), reference[item['reference_key']])
                      for value, item in zip(actual, OUTPUTS, strict=True)}
        if not all(v['elementwise_gate_passed'] for v in checks.values()):
            raise ValueError('Official depth/raw-pose wrapper differs from original Net')
        report.update(wrapper_CPU_seconds=wrapper_seconds, wrapper_CPU_comparisons=checks)
        onnx_path = args.output/'model.onnx'
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            tick = time.perf_counter()
            torch.onnx.export(wrapper, (inputs[0],), str(onnx_path), export_params=True,
                              opset_version=17, do_constant_folding=True, input_names=['images'],
                              output_names=[v['name'] for v in OUTPUTS], dynamic_axes=None,
                              training=torch.onnx.TrainingMode.EVAL, keep_initializers_as_inputs=False)
            report['export_seconds'] = time.perf_counter()-tick
            report['trace_warnings'] = [str(w.message)[:1000] for w in captured]
        graph = onnx.load(onnx_path, load_external_data=False)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError('Unexpected ONNX external weights; require one owned model file')
        onnx.checker.check_model(graph, full_check=True)
        def shape(v):
            return [d.dim_value for d in v.type.tensor_type.shape.dim]
        if (len(graph.graph.input) != 1 or graph.graph.input[0].name != 'images'
                or shape(graph.graph.input[0]) != list(SHAPE)
                or [v.name for v in graph.graph.output] != [v['name'] for v in OUTPUTS]
                or [shape(v) for v in graph.graph.output] != [v['shape'] for v in OUTPUTS]
                or any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                       for v in list(graph.graph.input)+list(graph.graph.output))):
            raise ValueError('Exported fixed input/output contract differs')
        report.update(onnx=dict(file='model.onnx', bytes=onnx_path.stat().st_size, sha256=sha(onnx_path)),
                      input=dict(name='images', shape=list(SHAPE), dtype='float32', layout='NCHW; N=view count'),
                      outputs=OUTPUTS, external_data_used=False,
                      original_weights_unchanged=weight_manifest(model) == before)
        if not report['original_weights_unchanged']:
            raise ValueError('Original model state changed during export')
    report['status'] = 'exported_not_runtime_validated'


def check_ort(args, report, pins):
    require_sha(args.export_receipt, args.export_sha256)
    exported = json.loads(args.export_receipt.read_text())
    if exported.get('status') != 'exported_not_runtime_validated' or exported.get('outputs') != OUTPUTS:
        raise ValueError('Successful matching full-depth/raw-pose export required')
    require_sha(__file__, exported['script_sha256'])
    pins.update(exported['source_and_input_hashes'])
    pins[str(args.export_receipt)] = args.export_sha256
    for path, digest in pins.items():
        require_sha(path, digest)
    if args.source_dir.resolve() != Path(exported['official_model_provenance']['source_dir']).resolve():
        raise ValueError('ORT official codec source must match the pinned export source directory')
    common = load_common(args)
    if pins.get(str(args.common)) != args.common_sha256 or pins.get(str(args.manifest)) != args.manifest_sha256:
        raise ValueError('ORT common/manifest SHA must match the original export pins')
    bind_source_pins(args, common, pins)
    common.install_official_import_path(args.source_dir)
    import torch
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = args.threads, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    directory = args.export_receipt.parent
    model_path = directory/exported['onnx']['file']
    require_sha(model_path, exported['onnx']['sha256'])
    pins[str(model_path)] = exported['onnx']['sha256']
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    ins, outs = session.get_inputs(), session.get_outputs()
    if (len(ins) != 1 or ins[0].name != 'images' or ins[0].shape != list(SHAPE) or ins[0].type != 'tensor(float)'
            or [(o.name, o.shape, o.type) for o in outs] != [(v['name'], v['shape'], 'tensor(float)') for v in OUTPUTS]):
        raise ValueError('ORT input/output names, shape or type mismatch')
    report.update(export_sha256=args.export_sha256, onnx=exported['onnx'], inputs=exported['input'], outputs=OUTPUTS,
                  onnxruntime_version=ort.__version__, CPUExecutionProvider_only=True,
                  graph_optimization='ORT_DISABLE_ALL', pairs=[], artifacts={})
    for pair in exported['pairs']:
        i = pair['index']
        values = {}
        for key in ('input_artifact', 'reference_artifact'):
            name = pair[key]
            path = directory/name
            require_sha(path, exported['artifacts'][name]['sha256'])
            pins[str(path)] = exported['artifacts'][name]['sha256']
            values[key] = path
        images = np.load(values['input_artifact'], allow_pickle=False)
        if record(images) != exported['artifacts'][pair['input_artifact']]['array']:
            raise ValueError('ORT input raw metadata differs')
        tick = time.perf_counter()
        actual = session.run([v['name'] for v in OUTPUTS], {'images': images})
        seconds = time.perf_counter()-tick
        with np.load(values['reference_artifact'], allow_pickle=False) as reference:
            if set(reference.files) != {'depth', 'conf', 'pose', 'extrinsic', 'intrinsic'}:
                raise ValueError('Complete original reference arrays required')
            if {k:record(reference[k]) for k in reference.files} != exported['artifacts'][pair['reference_artifact']]['arrays']:
                raise ValueError('ORT original reference raw metadata differs')
            checks = {v['name']: compare(a, reference[v['reference_key']]) for a,v in zip(actual, OUTPUTS, strict=True)}
            E, K, codec_seconds = official_codec(actual[2])
            codec = dict(extrinsic=compare(E, reference['extrinsic']), intrinsic=compare(K, reference['intrinsic']))
        name = f'actual-pair-{i}.npz'
        saved = {v['reference_key']:a for v,a in zip(OUTPUTS, actual, strict=True)}
        saved.update(extrinsic=E, intrinsic=K)
        with (args.output/name).open('xb') as stream:
            np.savez(stream, **saved)
        report['artifacts'][name] = dict(file=name, bytes=(args.output/name).stat().st_size,
                                        sha256=sha(args.output/name), arrays={k:record(v) for k,v in saved.items()})
        report['pairs'].append(dict(index=i, ORT_forward_seconds=seconds, official_codec_seconds=codec_seconds,
                                    comparisons=checks, codec_comparisons=codec))
    report['official_import_origins'] = common.check_import_origins(args.source_dir, exported['official_model_provenance']['source_hashes'])
    report['exact_graph_reference_gate_passed'] = all(
        v['elementwise_gate_passed'] for p in report['pairs'] for group in ('comparisons', 'codec_comparisons') for v in p[group].values())
    if len(report['pairs']) != 3 or not report['exact_graph_reference_gate_passed']:
        raise ValueError('Full three-pair ORT+official-codec strict gate failed; do not compile')
    report['status'] = 'completed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('export', 'ort'), required=True)
    parser.add_argument('--common', type=Path, default=Path(__file__).with_name('da3_probe_common.py'))
    parser.add_argument('--common-sha256')
    parser.add_argument('--source-dir', type=Path, default=Path(__file__).with_name('official'))
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('manifest.json'))
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--rgb-inputs', type=Path, nargs=3)
    parser.add_argument('--rgb-sha256', nargs=3)
    parser.add_argument('--export-receipt', type=Path)
    parser.add_argument('--export-sha256')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cpus', default='4,5,6,7')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    compatible = require_board()
    output = args.output.resolve()
    if not output.is_relative_to(BASE/'experiments') or output == BASE/'experiments':
        parser.error('Use a new output below board experiments')
    protected = [args.common, args.source_dir, args.manifest, *(args.rgb_inputs or [])]
    protected += [p for p in (args.checkpoint, args.export_receipt) if p is not None]
    if any(output == p.resolve() or output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        parser.error('Output must not overlap source/input artifacts')
    output.mkdir(parents=True, exist_ok=False)
    original_affinity = os.sched_getaffinity(0)
    start = time.monotonic()
    pins = {str(Path(__file__).resolve()): sha(__file__)}
    report = dict(kind='DA3_SMALL_complete_learned_depth_raw_pose_ONNX_probe', status='failed', mode=args.mode,
        script_sha256=sha(__file__), device_tree_compatible=compatible, model_name='da3-small',
        original_VGGT_weights_used=False, different_model_from_VGGT=True, original_DA3_weights=True,
        production_changed=False, ORB_used=False, GT_used=False, host_model_inference_executed=False,
        npu_used=False, all_operators_on_npu_verified=False, quality_validated=False, realtime_validated=False,
        auxiliary_ray_outputs_discarded_by_original_default=True,
        ONNX_dead_auxiliary_operations_may_be_pruned=True, pose_codec_runs_on_CPU=True, cleanup_errors=[])
    try:
        cpus = {int(v) for v in args.cpus.split(',')}
        if not cpus or not cpus <= original_affinity or not 1 <= args.threads <= 4:
            raise ValueError('Valid available CPU affinity and 1..4 threads required')
        os.sched_setaffinity(0, cpus)
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        report.update(cpu_affinity=sorted(cpus), torch_threads=torch.get_num_threads(),
                      torch_interop_threads=torch.get_num_interop_threads(),
                      process_wait_policy={key:os.environ.get(key) for key in ('OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')})
        (export_model if args.mode == 'export' else check_ort)(args, report, pins)
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        report['source_and_input_hashes'] = pins
        try:
            report['sources_and_inputs_unchanged'] = all(sha(p) == h for p,h in pins.items())
            if not report['sources_and_inputs_unchanged']:
                report['cleanup_errors'].append('Source/input SHA changed')
        except Exception as error:
            report['sources_and_inputs_unchanged'] = False
            report['cleanup_errors'].append(f'Source/input SHA verification failed: {error}')
        try:
            os.sched_setaffinity(0, original_affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
        except Exception as error:
            report['cleanup_errors'].append(str(error))
        report['total_seconds'] = time.monotonic()-start
        if report['cleanup_errors']:
            report['status'] = 'failed'
        receipt = output/('export.json' if args.mode == 'export' else 'ort.json')
        receipt.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], error=report.get('error'), output=str(output),
                          receipt_sha256=sha(receipt))), flush=True)
    return 0 if report['status'] in ('completed', 'exported_not_runtime_validated') else 1


if __name__ == '__main__':
    raise SystemExit(main())
