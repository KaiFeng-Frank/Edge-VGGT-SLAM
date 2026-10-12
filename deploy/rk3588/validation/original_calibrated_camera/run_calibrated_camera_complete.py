#!/usr/bin/env python3
"""Scoped original INT8-aggregator and INT8-CameraHead complete comparator.

Pass the frozen original FP16 retained manifest to the comparator after --;
the separate quantized mapping only selects four newly compiled RKNN files.
DINO/depth, original SL4, inputs, FP32 reference gates and pageout remain
in the original entrypoint. Immutable execution is explicit and defaults off.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
ENTRY = 'deploy/rk3588/validation/original_full_aggregator/compare_complete_model.py'
ENTRY_SHA = '75b81c6b8dd2c9fcb96f76bdb5fecb5d15e5c7794a161deee8aa4f323d020e7c'
CACHE_SHA = '3a8cc8c5a68c3bb934969e86053538f0d2026da1846274f4e6085297474a1e4b'
POLICY_SHA = 'a731fb1f26ea82f31fa7123792625e1731e19d3e35723930aef18e331dbe6897'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter-source', type=Path, required=True)
    p.add_argument('--adapter-sha256', required=True)
    p.add_argument('--quantized-manifest', type=Path, required=True)
    p.add_argument('--quantized-manifest-sha256', required=True)
    p.add_argument('--camera-adapter-source', type=Path, required=True)
    p.add_argument('--camera-adapter-sha256', required=True)
    p.add_argument('--camera-compiled', type=Path, required=True)
    p.add_argument('--camera-compiled-sha256', required=True)
    p.add_argument('--immutable-execution', action='store_true')
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args()
    child = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    required = ('--retained-aggregator', '--paired-aggregator', '--half-io-library',
                '--dino-manifest', '--grouped-dino', '--depth-manifest', '--reference-run', '--camera-split4')
    if any(flag not in child for flag in required):
        p.error('The complete original retained/grouped-DINO/depth/half-IO/reference branch is required')
    if '--output' not in child or child.count('--output') != 1:
        p.error('One original comparator --output path is required')
    output = Path(child[child.index('--output') + 1])
    if args.receipt.exists() or args.receipt.resolve().is_relative_to(output.resolve()):
        p.error('Use a new wrapper receipt outside the comparator output directory')
    if b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        raise RuntimeError('RK3588 board execution only')
    entry = BASE / 'src' / ENTRY
    if (sha(entry) != ENTRY_SHA or sha(args.adapter_source) != args.adapter_sha256
            or sha(args.quantized_manifest) != args.quantized_manifest_sha256
            or sha(BASE / 'src/vggt_slam/patch_cache.py') != CACHE_SHA
            or sha(args.camera_adapter_source) != args.camera_adapter_sha256
            or sha(args.camera_compiled) != args.camera_compiled_sha256
            or sha(BASE / 'src/vggt_slam/rknn_camera_head.py') != 'e20518d08c47746e1bf58e07bd6d0270bc7c7f6daa4d0858593f478ce6dfb09e'):
        raise ValueError('Frozen comparator/cache/adapter/mapping source identity differs')
    sys.path.insert(0, str(BASE / 'src'))
    spec = importlib.util.spec_from_file_location('_v80_calibrated_retained', args.adapter_source)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    import vggt_slam.rknn_retained_aggregator as retained_module
    import vggt_slam.patch_cache as cache_module
    import vggt_slam.rknn_camera_head as camera_module
    camera_spec = importlib.util.spec_from_file_location('_v82_calibrated_camera', args.camera_adapter_source)
    camera_adapter = importlib.util.module_from_spec(camera_spec)
    sys.modules[camera_spec.name] = camera_adapter
    camera_spec.loader.exec_module(camera_adapter)
    policy = None
    if args.immutable_execution:
        if sha(BASE / 'src/vggt_slam/immutable_execution_session.py') != POLICY_SHA:
            raise ValueError('Frozen immutable-session policy differs')
        from vggt_slam import immutable_execution_session as policy
    original_retained = retained_module.install_retained_aggregator
    original_cache = cache_module.install_patch_cache
    original_camera = camera_module.install_rknn_camera_head
    camera_handles, camera_failures = [], []
    records = []
    report = dict(schema='v82_original_VGGT_calibrated_camera_complete_entrypoint',
        status='running', original_model_only=True, quality_accepted=False,
        strict_rtol=1e-4, strict_atol=1e-5, original_model_math_and_parameters_unchanged=True,
        compiled_aggregator_internal_precision_changed=True,
        original_aggregator_ONNX_unchanged=True, quantized_dtype='w8a8',
        DINO_depth_precision_unchanged=True, compiled_camera_internal_precision_changed=True,
        original_camera_ONNX_unchanged=True, requested_camera_quantized_dtype='w8a8',
        camera_adapter_source_sha256=args.camera_adapter_sha256, camera_compile_receipt_sha256=args.camera_compiled_sha256,
        immutable_execution=args.immutable_execution,
        original_mutable_guard_semantics_preserved=not args.immutable_execution,
        entrypoint_sha256=ENTRY_SHA, adapter_source_sha256=args.adapter_sha256,
        quantized_manifest_sha256=args.quantized_manifest_sha256,
        script_sha256=sha(__file__), sessions=[], cleanup_errors=[])
    old_argv, started = sys.argv, time.monotonic()

    def install_retained(model, manifest, digest, **options):
        return adapter.install_calibrated_retained(model, manifest, digest,
            quantized_manifest=args.quantized_manifest,
            quantized_manifest_sha256=args.quantized_manifest_sha256, **options)

    def install_camera(model, **options):
        try:
            handle = camera_adapter.install_calibrated_camera(model, args.camera_compiled,
                args.camera_compiled_sha256, io_helper_source=args.adapter_source, **options)
            camera_handles.append(handle)
            return handle
        except BaseException as exc:
            if hasattr(exc, 'camera_rknn_metrics'):
                camera_failures.append(exc.camera_rknn_metrics)
            raise

    def install_cache(model, *positional, **options):
        cache = original_cache(model, *positional, **options)
        if policy is None:
            return cache
        session = None
        try:
            session = policy.install_immutable_session(model, cache,
                model.aggregator._vggt_rknn_blocks_owner,
                model.aggregator.patch_embed._vggt_rknn_dino_owner,
                contract='exclusive_immutable_model')
            close, metrics = cache.close, cache.metrics
            record = dict(cache=cache, session=session, close=close, metrics=metrics, closing=False)
            records.append(record)
            def close_session_cache():
                if record['closing']:
                    return close()
                record['closing'] = True
                try:
                    session.close()
                finally:
                    close()
            def cache_metrics():
                result = metrics()
                result['immutable_execution_session'] = session.metrics()
                return result
            cache.close, cache.metrics = close_session_cache, cache_metrics
            return cache
        except BaseException:
            try:
                if session is not None:
                    session.close()
            finally:
                cache.close()
            raise

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        retained_module.install_retained_aggregator = install_retained
        cache_module.install_patch_cache = install_cache
        camera_module.install_rknn_camera_head = install_camera
        sys.argv = [str(entry), *child]
        try:
            runpy.run_path(str(entry), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
        report['status'] = 'completed_original_entrypoint_pending_saved_output_review'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        for record in reversed(records):
            try:
                record['cache'].close()
            except BaseException as exc:
                report['cleanup_errors'].append(type(exc).__name__ + ': ' + str(exc))
            report['sessions'].append(record['session'].metrics())
            for name in ('close', 'metrics'):
                if name in record['cache'].__dict__:
                    delattr(record['cache'], name)
        for handle in reversed(camera_handles):
            try:
                handle.close()
            except BaseException as exc:
                report['cleanup_errors'].append('CameraHead: ' + type(exc).__name__ + ': ' + str(exc))
        retained_module.install_retained_aggregator = original_retained
        cache_module.install_patch_cache = original_cache
        camera_module.install_rknn_camera_head = original_camera
        report['camera_rknn'] = [handle.metrics() for handle in camera_handles]
        report['camera_installation_failures'] = camera_failures
        report['camera_resources_closed'] = all(row['closed'] and row['live_contexts'] == 0
            and row['contexts_created'] == row['contexts_destroyed'] and row['cleanup_errors'] == []
            for row in report['camera_rknn'])
        sys.argv = old_argv
        report.update(installers_restored=(retained_module.install_retained_aggregator is original_retained
            and cache_module.install_patch_cache is original_cache
            and camera_module.install_rknn_camera_head is original_camera),
            sources_unchanged=(sha(entry) == ENTRY_SHA and sha(args.adapter_source) == args.adapter_sha256
                and sha(args.quantized_manifest) == args.quantized_manifest_sha256
                and sha(args.camera_adapter_source) == args.camera_adapter_sha256
                and sha(args.camera_compiled) == args.camera_compiled_sha256),
            seconds=time.monotonic() - started)
        if (report['cleanup_errors'] or not report['sources_unchanged'] or not report['installers_restored']
                or not report['camera_resources_closed']):
            report['status'] = 'failed_cleanup_or_identity'
        # Original branch names remain stable for saved-array comparison; make
        # their historical FP16 spelling explicitly non-authoritative here.
        result_path = output / 'result.json'
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            result.update(internal_precision_changed=True,
                calibrated_retained_aggregator=True, requested_aggregator_quantized_dtype='w8a8',
                original_aggregator_ONNX_unchanged=True,
                legacy_branch_names_do_not_describe_aggregator_precision=True,
                calibrated_entrypoint_receipt_sha256=None,
                quantized_manifest_sha256=args.quantized_manifest_sha256,
                calibrated_camera_head=True, requested_camera_quantized_dtype='w8a8',
                original_camera_ONNX_unchanged=True,
                legacy_branch_names_do_not_describe_camera_precision=True,
                camera_compile_receipt_sha256=args.camera_compiled_sha256)
            result.setdefault('runtime_source_sha256', {})['private/rknn_calibrated_retained.py'] = args.adapter_sha256
            result['runtime_source_sha256']['private/rknn_calibrated_camera.py'] = args.camera_adapter_sha256
            result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        args.receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: report.get(k) for k in ('status', 'cleanup_errors')}, allow_nan=False), flush=True)
    return 0 if report['status'].startswith('completed_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
