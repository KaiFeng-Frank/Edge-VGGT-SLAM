#!/usr/bin/env python3
"""Board-only original GPU staging tail on saved actual callback operands.

This diagnosis includes snapshots and readback; it makes no performance or
accuracy acceptance claim. It never loads model weights, Torch, ORT or RKNN.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path('/mnt/vggt-data/vggt-slam2')
LEGACY_SHA = 'f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2'
COMPONENT_SHA = '9dc9e66b9b0521ff168e31063c47bd21c6dc4cbccc9bd339a4ecf28b647b33e7'
MALI_SHA = 'd57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, digest, name):
    require(sha(path) == digest, 'Frozen source/library changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(np, actual, reference):
    require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32,
            'Compared callback arrays must share original FP32 shape')
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    d = a - b
    outside = int(np.count_nonzero(np.abs(d) > 1e-5 + 1e-4 * np.abs(b)))
    return dict(strict_rtol=1e-4, strict_atol=1e-5, strict_pass=outside == 0,
        outside_tolerance=outside, elements=actual.size,
        byte_equal=actual.tobytes() == reference.tobytes(), max_abs=float(np.abs(d).max()),
        relative_l2=float(np.linalg.norm(d.ravel()) / max(np.linalg.norm(b.ravel()), 1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-domain', choices=('cpu','gpu'), default='gpu',
                        help='GPU readback is authoritative for actual borrowed-kernel inputs; CPU snapshots may precede queued SDK preparation')
    parser.add_argument('--component-source', type=Path,
                        default=Path(__file__).with_name('original_attention_tail_opencl.py'))
    parser.add_argument('--legacy-source', type=Path, default=BASE / 'src/deploy/rk3588/validation/original_native_gpu/native_erf_gelu_opencl.py')
    parser.add_argument('--library', type=Path, default=BASE / 'experiments/libmali-valhall-g610-g6p0-dummy.so')
    args = parser.parse_args()
    legacy = load(args.legacy_source, LEGACY_SHA, '_v83_callback_shadow_legacy')
    legacy._board_guard()
    component = load(args.component_source, COMPONENT_SHA, '_v83_callback_shadow_component')
    require(sha(args.library) == MALI_SHA, 'Original private board OpenCL library changed')
    require(sha(args.capture) == args.capture_sha256, 'Saved callback receipt changed')
    captured = json.loads(args.capture.read_text())
    require(captured['schema'] == 'v83_original_attention_custom_callback_diagnostic'
            and captured['captures'], 'Actual callback captures required')
    import numpy as np
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v83_original_attention_callback_staging_shadow', status='failed',
        script_sha256=sha(__file__), capture_sha256=args.capture_sha256,
        original_model_only=True, original_kernel_source_sha256=COMPONENT_SHA,
        no_RKNN_or_model_inference=True, normal_performance_claimed=False,
        candidate_accepted=False, input_domain=args.input_domain,
        CPU_snapshot_assumed_current=False, comparisons=[], cleanup_errors=[])
    bridges, fixed = {}, {}
    Bridge = component.make_bridge_class(legacy)
    try:
        for row in captured['captures']:
            require(row['status'] == 'returned' and row['GPU_readback_completed'], 'Complete CPU/GPU callback readback required')
            artifact = row['artifact']
            path = (args.capture.parent / artifact['file']).resolve(strict=True)
            require(path.is_relative_to(args.capture.parent.resolve())
                    and path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256'], 'Callback NPZ identity changed')
            with np.load(path, allow_pickle=False) as archive:
                require(set(archive.files) == set(artifact['arrays']), 'Callback NPZ array catalog differs')
                arrays = {name: archive[name].copy(order='C') for name in archive.files}
            for name, value in arrays.items():
                meta = artifact['arrays'][name]
                require(value.dtype == np.float32 and list(value.shape) == meta['shape']
                    and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == meta['raw_sha256'], 'Callback array identity differs: ' + name)
            cpu = [arrays['CPU_input_' + str(i).zfill(2)] for i in range(11)]
            inputs = ([arrays['GPU_' + name] for name in ('q','k','v')]
                      if args.input_domain == 'gpu' else cpu[:3])
            role = row['role']
            b, h, s, d = inputs[0].shape
            require((b,h,s,d) in ((2,16,53,64),(1,16,106,64)), 'Original full attention shape required')
            require(row['epsilon'] == 9.999999747378752e-06, 'Original FP32 epsilon differs')
            constants = [cpu[i] for i in range(3,7)]
            tables = [cpu[8][...,:32].copy(),cpu[9][...,:32].copy(),cpu[8][...,32:].copy(),cpu[9][...,32:].copy()]
            identity = [hashlib.sha256(x.tobytes()).hexdigest() for x in constants + tables]
            if role not in bridges:
                bridge = Bridge.__new__(Bridge)
                bridges[role] = bridge
                bridge.__init__(shape=(b,h,s,d), parameters=constants, tables=tables,
                    q_eps=1e-5, k_eps=1e-5, scale=float(cpu[10].item()), library=str(args.library))
                fixed[role] = identity
            require(identity == fixed[role], 'Same-role original callback constants changed')
            output, middle, timing = bridges[role].run(*inputs, capture_intermediates=True)
            name = role + '-' + str(row['role_callback_index']).zfill(2)
            saved = args.output_dir / (name + '-staging.npz')
            np.savez(saved, proj_input=output, **middle)
            checks = {name: compare(np, arrays['GPU_' + name], cpu[i])
                      for i,name in enumerate(('q','k','v'))}
            checks.update(
                borrowed_GPU_output_vs_staging=compare(np, arrays['GPU_proj_input'], output),
                CPU_after_FROM_vs_GPU_readback=compare(np, arrays['CPU_output_after_FROM_DEVICE'], arrays['GPU_proj_input']),
                CPU_after_FROM_vs_staging=compare(np, arrays['CPU_output_after_FROM_DEVICE'], output))
            checks.update({name + '_borrowed_vs_staging': compare(np, arrays['GPU_' + name], value)
                           for name,value in middle.items()})
            report['comparisons'].append(dict(role=role, role_callback_index=row['role_callback_index'],
                checks=checks, input_domain=args.input_domain,
                staging_input_raw_sha256=[hashlib.sha256(value.tobytes()).hexdigest() for value in inputs],
                staging_diagnostic_timing=timing,
                artifact=dict(file=saved.name,bytes=saved.stat().st_size,sha256=sha(saved),
                    arrays={key:dict(shape=list(value.shape),dtype=str(value.dtype),
                        raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
                        for key,value in dict(proj_input=output,**middle).items()})))
        report['status'] = 'diagnostic_completed_no_quality_acceptance'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for role,bridge in reversed(list(bridges.items())):
            try:
                bridge.close()
                require(bridge.metrics()['fully_released'], 'Staging GPU cleanup incomplete')
            except BaseException as exc:
                report['cleanup_errors'].append(role + ': ' + str(exc))
        report['GPU_lifecycle'] = {role:bridge.metrics() for role,bridge in bridges.items()}
        report['source_and_capture_unchanged'] = all(sha(path) == digest for path,digest in (
            (args.capture,args.capture_sha256),(args.component_source,COMPONENT_SHA),
            (args.legacy_source,LEGACY_SHA),(args.library,MALI_SHA)))
        if report['cleanup_errors'] or not report['source_and_capture_unchanged']:
            report['status'] = 'failed_cleanup_or_identity'
        (args.output_dir / 'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({key:report.get(key) for key in ('status','error','cleanup_errors')}),flush=True)
    return 0 if report['status'] == 'diagnostic_completed_no_quality_acceptance' else 1


if __name__ == '__main__':
    raise SystemExit(main())
