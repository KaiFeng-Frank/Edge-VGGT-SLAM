#!/usr/bin/env python3
"""Board-only original-weight DPVO encoder capture, export and CPU ORT probe.

These are the two original CNNs, not a complete VO or a replacement for VGGT.
All preprocessing, real-checkpoint forwards and tracing run on the RK3588S.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def meta(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    return dict(shape=list(array.shape), dtype=str(array.dtype), bytes=array.nbytes,
                tensor_sha256=hashlib.sha256(array.tobytes()).hexdigest())


def file_meta(path):
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def compare(actual, reference):
    import numpy as np
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('Output shape/dtype mismatch')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite output')
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    failures = np.abs(diff) > (1e-5 + 1e-4 * np.abs(reference.astype(np.float64)))
    return dict(rtol=1e-4, atol=1e-5, failed_elements=int(failures.sum()),
                max_abs=float(np.max(np.abs(diff))),
                relative_l2=float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(reference.ravel().astype(np.float64)), 1e-30)),
                bitwise_equal=actual.tobytes() == reference.tobytes(),
                gate_passed=not bool(failures.any()))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['capture', 'export', 'ort'])
    for name in ('common', 'cpu-ops', 'source-dir', 'source-manifest', 'checkpoint', 'input-manifest', 'dataset-dir', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('common', 'cpu-ops', 'source-manifest', 'checkpoint', 'input-manifest'):
        p.add_argument('--'+name+'-sha256', required=True)
    p.add_argument('--capture-dir', type=Path)
    p.add_argument('--capture-sha256')
    p.add_argument('--export-dir', type=Path)
    p.add_argument('--export-sha256')
    args = p.parse_args(argv)
    if platform.machine() not in ('aarch64', 'arm64'):
        p.error('Real-model work is board-only')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pins = {}
    def pin(path, expected):
        actual = sha(path)
        if actual != expected:
            raise ValueError('Pinned file SHA mismatch: '+str(path))
        pins[str(path.resolve())] = actual
    report = dict(schema='v26_DPVO_original_encoders_board_probe', mode=args.mode, status='failed',
        script_sha256=sha(__file__), machine=platform.machine(), python=sys.version,
        actual_inference_on_RK3588S=True, production_changed=False,
        encoder_only=True, complete_VO_executed=False, realtime_validated=False,
        checkpoint_sha256=args.checkpoint_sha256, input_manifest_sha256=args.input_manifest_sha256,
        calls=[], cleanup_errors=[], source_pins=pins)
    prior_affinity = os.sched_getaffinity(0)
    error = None
    try:
        for name in ('common', 'cpu_ops', 'source_manifest', 'checkpoint', 'input_manifest'):
            pin(getattr(args, name), getattr(args, name+'_sha256'))
        pin(Path(__file__), report['script_sha256'])
        source_manifest = json.loads(args.source_manifest.read_text())
        for item in source_manifest['retrieved_files']:
            if 'sha256' in item:
                pin(args.source_dir/item['path'], item['sha256'])
        manifest = json.loads(args.input_manifest.read_text())
        pin(args.dataset_dir/'rgb.txt', manifest['rgb_index_sha256'])
        for row in manifest['frames']:
            path = args.dataset_dir/row['relative_path']
            if path.stat().st_size != row['bytes']:
                raise ValueError('Input image size mismatch')
            pin(path, row['sha256'])
        os.sched_setaffinity(0, {4,5,6,7})
        import numpy as np
        import torch
        import cv2
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(1)
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(),
                      versions=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),
                      preprocessing=manifest['preprocessing'])
        common = load_module(args.common, '_v26_dpvo_common')
        started = time.monotonic()
        neural, provenance = common.load_neural(args.source_dir, args.source_manifest, args.checkpoint,
            args.checkpoint_sha256, args.cpu_ops, cpu_ops_sha256=args.cpu_ops_sha256)
        neural.eval().requires_grad_(False)
        report.update(loader=provenance, load_seconds=time.monotonic()-started)
        encoder = common.make_encoder(neural).eval()
        class FixedOneViewEncoder(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def forward(self, normalized_bgr):
                fmap, imap = self.original(normalized_bgr[:, None])
                return fmap[:, 0], imap[:, 0]
        flat_encoder = FixedOneViewEncoder(encoder).eval()
        before_weights = {k: meta(v.detach().cpu().numpy()) for k,v in neural.state_dict().items()}
        report['state_before'] = before_weights
        if args.mode == 'capture':
            images = []
            report['inputs'] = []
            K = np.array([[517.3,0,318.6],[0,516.5,255.3],[0,0,1]], dtype=np.float64)
            D = np.array([0.2624,-0.9531,-0.0054,0.0026,1.1633], dtype=np.float64)
            for i,row in enumerate(manifest['frames']):
                t = time.monotonic()
                bgr = cv2.imread(str(args.dataset_dir/row['relative_path']))
                if bgr is None or bgr.shape != (480,640,3) or bgr.dtype != np.uint8:
                    raise ValueError('Expected original TUM 640x480 BGR image')
                cropped = np.ascontiguousarray(cv2.undistort(bgr,K,D)[8:-8,16:-16].transpose(2,0,1))
                image = 2 * (torch.from_numpy(cropped)[None,None] / 255.0) - 0.5
                if image.shape != (1,1,3,464,608) or image.dtype != torch.float32:
                    raise ValueError('Original preprocessing shape/dtype mismatch')
                preprocessing_seconds = time.monotonic()-t
                path = args.output_dir/f'input-{i}.npy'
                np.save(path, image.numpy(), allow_pickle=False)
                report['inputs'].append(dict(frame_index=i, frame_id=row['frame_id'], source=row,
                    preprocessing_seconds=preprocessing_seconds, input=file_meta(path),
                    array=meta(image.numpy()), cropped_BGR_uint8=meta(cropped),
                    pixel_intrinsics=[517.3,516.5,302.6,247.3]))
                images.append(image)
            representative = {}
            with torch.inference_mode():
                for call_index, i in enumerate((0,0,0,0,1,2)):
                    t = time.monotonic()
                    f = neural.patchify.fnet(images[i]) / 4.0
                    ftime = time.monotonic()-t
                    t = time.monotonic()
                    im = neural.patchify.inet(images[i]) / 4.0
                    itime = time.monotonic()-t
                    outputs = dict(fmap=f.detach().numpy().copy(), imap=im.detach().numpy().copy())
                    if outputs['fmap'].shape != (1,1,128,116,152) or outputs['imap'].shape != (1,1,384,116,152):
                        raise ValueError('Original encoder output shape mismatch')
                    if any(not np.isfinite(a).all() or not np.any(a) for a in outputs.values()):
                        raise ValueError('Invalid encoder output')
                    arrays = {k:meta(v) for k,v in outputs.items()}
                    if i in representative and arrays != representative[i]:
                        raise ValueError('Repeated encoder output changed')
                    representative[i] = arrays
                    path = args.output_dir/f'call-{call_index:02d}-frame-{i}.npz'
                    np.savez(path, **outputs)
                    record = dict(call_index=call_index, frame_index=i, frame_id=manifest['frames'][i]['frame_id'],
                        timing_role='first_call' if call_index==0 else 'warm_same_frame' if call_index<4 else 'other_frame_single',
                        fnet_seconds=ftime, inet_seconds=itime, total_encoder_seconds=ftime+itime,
                        artifact=file_meta(path), arrays=arrays)
                    report['calls'].append(record)
                    print(json.dumps({k:v for k,v in record.items() if k not in ('arrays','artifact')}), flush=True)
            report.update(status='completed_encoder_cpu_probe_not_VO',
                first_frame_warm_mean_seconds=sum(r['total_encoder_seconds'] for r in report['calls'][1:4])/3,
                timing_scope='Each original CNN forward and its /4 scale. Input decoding/undistortion/crop/normalization separately timed; loading, copies for evidence, hashing and saving excluded. No patch correlation, update, BA or pose output.')
        else:
            if args.capture_dir is None or args.capture_sha256 is None:
                raise ValueError('Capture and SHA required')
            pin(args.capture_dir/'capture.json', args.capture_sha256)
            captured = json.loads((args.capture_dir/'capture.json').read_text())
            if captured['status'] != 'completed_encoder_cpu_probe_not_VO' or captured['checkpoint_sha256'] != args.checkpoint_sha256 or captured['input_manifest_sha256'] != args.input_manifest_sha256 or captured['state_before'] != before_weights:
                raise ValueError('Wrong completed CPU reference')
            if captured['script_sha256'] != report['script_sha256'] or any(
                captured['source_pins'].get(str(path.resolve())) != pins[str(path.resolve())]
                for path in (args.common, args.cpu_ops, args.source_manifest)):
                raise ValueError('CPU reference must use the same frozen probe and implementation')
            report['capture_sha256'] = args.capture_sha256
            inputs = []
            references = []
            for i,row in enumerate(captured['inputs']):
                pin(args.capture_dir/row['input']['file'], row['input']['sha256'])
                value = np.load(args.capture_dir/row['input']['file'], allow_pickle=False)
                if meta(value) != row['array']:
                    raise ValueError('Changed CPU input')
                inputs.append(torch.from_numpy(value))
                original = next(c for c in captured['calls'] if c['frame_index']==i)
                pin(args.capture_dir/original['artifact']['file'], original['artifact']['sha256'])
                with np.load(args.capture_dir/original['artifact']['file'], allow_pickle=False) as saved:
                    references.append({k:saved[k].copy() for k in ('fmap','imap')})
            if args.mode == 'export':
                with torch.inference_mode():
                    report['wrapper_checks'] = []
                    for i,x in enumerate(inputs):
                        actual = flat_encoder(x[:, 0])
                        checks={k:compare(y.numpy()[:,None],references[i][k]) for k,y in zip(('fmap','imap'),actual)}
                        report['wrapper_checks'].append(checks)
                        if not all(c['gate_passed'] for c in checks.values()):
                            raise ValueError('Encoder wrapper numerical gate failed')
                    target = args.output_dir/'encoders.onnx'
                    t=time.monotonic()
                    torch.onnx.export(flat_encoder,inputs[0][:,0],str(target),input_names=['normalized_bgr'],
                        output_names=['fmap','imap'],opset_version=17,do_constant_folding=True,
                        training=torch.onnx.TrainingMode.EVAL)
                    report['export_seconds']=time.monotonic()-t
                import onnx
                onnx.checker.check_model(str(target),full_check=True)
                report.update(status='exported_not_runtime_validated', onnx=file_meta(target),
                    input_shape=[1,3,464,608], output_shapes=[[1,128,116,152],[1,384,116,152]],
                    adapter='Fixed one view: insert singleton view axis before official encoders; remove only that axis in outputs')
            else:
                if args.export_dir is None or args.export_sha256 is None:
                    raise ValueError('Export and SHA required')
                pin(args.export_dir/'export.json',args.export_sha256)
                exported=json.loads((args.export_dir/'export.json').read_text())
                if exported['status']!='exported_not_runtime_validated' or exported['capture_sha256']!=args.capture_sha256:
                    raise ValueError('Wrong export reference')
                onnx_path=args.export_dir/exported['onnx']['file']
                pin(onnx_path,exported['onnx']['sha256'])
                import onnxruntime as ort
                options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
                options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
                report.update(onnx=exported['onnx'],export_sha256=args.export_sha256,ORT_version=ort.__version__,providers=session.get_providers(),graph_optimization='ORT_DISABLE_ALL')
                report['pairs']=[]
                for i,x in enumerate(inputs):
                    t=time.monotonic();raw=session.run(['fmap','imap'],{'normalized_bgr':x.numpy()[:,0]});seconds=time.monotonic()-t
                    actual=[y[:,None] for y in raw]
                    checks={k:compare(y,references[i][k]) for k,y in zip(('fmap','imap'),actual)}
                    path=args.output_dir/f'frame-{i}.npz';np.savez(path,**dict(zip(('fmap','imap'),actual)))
                    report['pairs'].append(dict(frame_index=i,forward_seconds=seconds,checks=checks,artifact=file_meta(path),arrays={k:meta(v) for k,v in zip(('fmap','imap'),actual)}))
                    if not all(c['gate_passed'] for c in checks.values()):
                        raise ValueError('Board CPU ORT numerical gate failed')
                report.update(status='completed_CPU_ORT_gate',exact_reference_gate_passed=True)
        report['state_unchanged']=before_weights=={k:meta(v.detach().cpu().numpy()) for k,v in neural.state_dict().items()}
        if not report['state_unchanged']:
            raise ValueError('Original parameters changed')
    except BaseException as exc:
        error=exc
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    finally:
        try:
            report['source_hashes_after']={name:sha(Path(name)) for name in pins}
            report['sources_and_inputs_unchanged']=report['source_hashes_after']==pins
            if not report['sources_and_inputs_unchanged']:
                raise ValueError('Pinned input/source changed')
        except Exception as exc:
            report['cleanup_errors'].append(repr(exc));error=error or exc
        try:
            os.sched_setaffinity(0,prior_affinity)
            report['affinity_restored']=os.sched_getaffinity(0)==prior_affinity
        except Exception as exc:
            report['cleanup_errors'].append(repr(exc));error=error or exc
        report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        receipt=args.output_dir/(args.mode+'.json')
        receipt.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],error=report.get('error'),receipt=str(receipt),receipt_sha256=sha(receipt))),flush=True)
    return 1 if error else 0


if __name__=='__main__':
    raise SystemExit(main())
