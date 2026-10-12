#!/usr/bin/env python3
"""Original INT8-aggregator and INT8-CameraHead timestamp replay experiment.

The child arguments select its existing single-model immutable/half-IO/pageout
branch. Source pacing, queue policy, predictions, SL4, publication and per-pose
clocks remain inside the unchanged replay script. Retained and CameraHead
installers are temporarily replaced. No model is imported when this source is inspected.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import runpy
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
ENTRY = 'scripts/replay_original_vggt.py'
FROZEN = {
    ENTRY: 'ed7ae7ba73d4e54d17a1be9bcc0c9213776371cbe90c5921210779ca6d252d3e',
    'vggt_slam/rknn_retained_aggregator.py': '04fbb5b2ad7db22bf26d52d4a6b5a3fadffa042557919575938b035f5dd06b5f',
    'vggt_slam/patch_cache.py': '3a8cc8c5a68c3bb934969e86053538f0d2026da1846274f4e6085297474a1e4b',
    'vggt_slam/immutable_execution_session.py': 'a731fb1f26ea82f31fa7123792625e1731e19d3e35723930aef18e331dbe6897',
    'vggt_slam/timestamp_replay.py': '14b168028a6f7722f4bea036285b4f9cd8119fed59230b7e3e07c4b7af07addc',
    'vggt_slam/rknn_grouped_dino.py': 'e0e5664dc231a9960db12ae6c690d3f0604417a4d74991dcbd0ca2ca765ef6ac',
    'vggt_slam/rknn_camera_head.py': 'e20518d08c47746e1bf58e07bd6d0270bc7c7f6daa4d0858593f478ce6dfb09e',
    'vggt_slam/rknn_depth_head.py': '7efacdb27be4aeb134fd2b64e2819136320fc45d25973517584522d314ba7894',
    'vggt_slam/solver.py': '33b5b4d816cd08f0cb9857f89c9e3fc770ad0c2ab3c9dc938864f00cefe86905',
    'vggt_slam/graph.py': 'dda8fb7c947c3c39baa7d190e2942cd4b9f26fe1557d7605fdc8651f27a5ab48',
    'vggt_slam/submap.py': 'fddccafee4e9f32452ce044b80669bd48f5fe5f331a980986a6091e7fce92e63',
    'vggt_slam/edge_mapping.py': '40cd678ccdc393f5f5aa7f08572e3cedceaca10712c59efd4dfce59c57605c47',
    'vggt_slam/model_execution.py': 'dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter-source', type=Path, required=True)
    parser.add_argument('--adapter-sha256', required=True)
    parser.add_argument('--quantized-manifest', type=Path, required=True)
    parser.add_argument('--quantized-manifest-sha256', required=True)
    parser.add_argument('--camera-adapter-source', type=Path, required=True)
    parser.add_argument('--camera-adapter-sha256', required=True)
    parser.add_argument('--camera-compiled', type=Path, required=True)
    parser.add_argument('--camera-compiled-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    child = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    flags = ('--rknn-retained-aggregator', '--rknn-paired-aggregator',
             '--rknn-grouped-dino', '--rknn-camera-head', '--rknn-camera-split4',
             '--checkpoint-pageout', '--immutable-execution')
    values = ('--rgb-index', '--checkpoint', '--output', '--rknn-manifest',
              '--rknn-manifest-sha256', '--half-io-library', '--half-io-library-sha256',
              '--rknn-dino-manifest', '--rknn-dino-manifest-sha256',
              '--rknn-depth-manifest', '--rknn-depth-manifest-sha256')
    if any(child.count(flag) != 1 for flag in flags + values) or '--no-patch-cache' in child:
        parser.error('Use the complete original single-model immutable/half-IO/pageout replay branch')
    for flag in values:
        i = child.index(flag)
        if i + 1 >= len(child) or child[i + 1].startswith('--'):
            parser.error('Missing separate value after ' + flag)
    output = Path(child[child.index('--output') + 1]).resolve()
    receipt = args.receipt.resolve()
    if receipt.exists() or receipt.is_relative_to(output):
        parser.error('Use a new wrapper receipt outside the original replay output directory')
    if output.exists():
        parser.error('The original replay output directory must be new')
    compatible = next((p.read_bytes() for p in (Path('/proc/device-tree/compatible'),
        Path('/sys/firmware/devicetree/base/compatible')) if p.is_file()), b'')
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or b'rk3588' not in compatible:
        raise RuntimeError('Actual execution is restricted to Linux RK3588/S')
    watched = {BASE / 'src' / name: digest for name, digest in FROZEN.items()}
    watched.update({args.adapter_source.resolve(): args.adapter_sha256,
                    args.quantized_manifest.resolve(): args.quantized_manifest_sha256,
                    args.camera_adapter_source.resolve(): args.camera_adapter_sha256,
                    args.camera_compiled.resolve(): args.camera_compiled_sha256})
    if any(sha(path) != digest for path, digest in watched.items()):
        raise ValueError('Frozen original replay/runtime/adapter/mapping identity differs')
    entry = BASE / 'src' / ENTRY
    sys.path.insert(0, str(BASE / 'src'))
    module_name = '_v82_calibrated_retained_replay'
    old_module = sys.modules.get(module_name)
    spec = importlib.util.spec_from_file_location(module_name, args.adapter_source)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = adapter
    spec.loader.exec_module(adapter)
    import vggt_slam.rknn_retained_aggregator as retained_module
    original_installer = retained_module.install_retained_aggregator
    import vggt_slam.rknn_camera_head as camera_module
    camera_module_name = '_v82_calibrated_camera_replay'
    old_camera_module = sys.modules.get(camera_module_name)
    camera_spec = importlib.util.spec_from_file_location(camera_module_name, args.camera_adapter_source)
    camera_adapter = importlib.util.module_from_spec(camera_spec)
    sys.modules[camera_module_name] = camera_adapter
    camera_spec.loader.exec_module(camera_adapter)
    original_camera = camera_module.install_rknn_camera_head
    camera_handles, camera_failures = [], []
    handles = []
    report = dict(schema='v82_original_VGGT_calibrated_camera_timestamp_replay_entrypoint',
        status='running', original_model_only=True, candidate_accepted=False,
        trajectory_quality_validated=False, real_time_validated=False,
        original_replay_entrypoint_unchanged=True, source_queue_and_pose_clocks_unchanged=True,
        original_aggregator_ONNX_unchanged=True, compiled_aggregator_internal_precision_changed=True,
        requested_aggregator_quantized_dtype='w8a8', all_internal_operators_INT8_verified=False,
        API_FP32_conversion_not_internal_FP32=True,
        DINO_depth_and_SL4_unchanged=True, compiled_camera_internal_precision_changed=True,
        original_camera_ONNX_unchanged=True, requested_camera_quantized_dtype='w8a8',
        camera_adapter_source_sha256=args.camera_adapter_sha256, camera_compile_receipt_sha256=args.camera_compiled_sha256,
        single_original_model_instance=True, immutable_execution=True,
        checkpoint_pageout=True, half_IO_for_DINO_unchanged=True,
        aggregator_half_IO_enabled=False, unsupported_loop_closure_disabled=True,
        entrypoint_sha256=FROZEN[ENTRY], frozen_runtime_source_sha256=FROZEN,
        adapter_source_sha256=args.adapter_sha256,
        quantized_manifest_sha256=args.quantized_manifest_sha256,
        script_sha256=sha(__file__), child_arguments=child,
        installer_calls=0, camera_installer_calls=0, aggregator_metrics_after_original_cleanup=[], cleanup_errors=[])
    old_argv, started = sys.argv, time.monotonic()

    def install_retained(model, manifest, digest, **options):
        report['installer_calls'] += 1
        if report['installer_calls'] != 1:
            raise RuntimeError('This wrapper permits one original model/aggregator installation')
        handle = adapter.install_calibrated_retained(model, manifest, digest,
            quantized_manifest=args.quantized_manifest,
            quantized_manifest_sha256=args.quantized_manifest_sha256, **options)
        handles.append(handle)
        return handle

    def install_camera(model, **options):
        report['camera_installer_calls'] += 1
        if report['camera_installer_calls'] != 1:
            raise RuntimeError('One original CameraHead installation is required')
        try:
            handle = camera_adapter.install_calibrated_camera(model, args.camera_compiled,
                args.camera_compiled_sha256, io_helper_source=args.adapter_source, **options)
            camera_handles.append(handle)
            return handle
        except BaseException as exc:
            if hasattr(exc, 'camera_rknn_metrics'):
                camera_failures.append(exc.camera_rknn_metrics)
            raise

    receipt.parent.mkdir(parents=True, exist_ok=True)
    write(receipt, report)
    try:
        retained_module.install_retained_aggregator = install_retained
        camera_module.install_rknn_camera_head = install_camera
        sys.argv = [str(entry), *child]
        try:
            runpy.run_path(str(entry), run_name='__main__')
        except SystemExit as error:
            if error.code not in (0, None):
                raise
        if report['installer_calls'] != 1 or len(handles) != 1:
            raise RuntimeError('The original replay did not install exactly one calibrated aggregator')
        if report['camera_installer_calls'] != 1 or len(camera_handles) != 1:
            raise RuntimeError('The original replay did not install exactly one calibrated CameraHead')
        report['status'] = 'completed_original_timestamp_replay_pending_quality_review'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        for handle in reversed(camera_handles):
            try:
                handle.close()
            except BaseException as error:
                report['cleanup_errors'].append('CameraHead: ' + type(error).__name__ + ': ' + str(error))
        report['camera_rknn'] = [handle.metrics() for handle in camera_handles]
        report['camera_installation_failures'] = camera_failures
        report['camera_resources_closed'] = all(row['closed'] and row['live_contexts'] == 0
            and row['contexts_created'] == row['contexts_destroyed'] and row['cleanup_errors'] == []
            for row in report['camera_rknn'])
        retained_module.install_retained_aggregator = original_installer
        camera_module.install_rknn_camera_head = original_camera
        sys.argv = old_argv
        if old_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_module
        if old_camera_module is None:
            sys.modules.pop(camera_module_name, None)
        else:
            sys.modules[camera_module_name] = old_camera_module
        report['installer_restored'] = (retained_module.install_retained_aggregator is original_installer
            and camera_module.install_rknn_camera_head is original_camera)
        report['sources_unchanged'] = all(sha(path) == digest for path, digest in watched.items())
        for handle in handles:
            metrics = handle.metrics()
            report['aggregator_metrics_after_original_cleanup'].append(metrics)
            if not metrics['closed'] or metrics['live_contexts'] or metrics['cleanup_errors']:
                report['cleanup_errors'].append('Original entrypoint did not cleanly close calibrated aggregator')
        # Only annotate precision/provenance after the original final publication
        # and cleanup. No source metrics, per-pose values or timing summaries change.
        result_path = output / 'results.json'
        if result_path.is_file():
            report['original_report_sha256_before_annotation'] = sha(result_path)
            result = json.loads(result_path.read_text())
            report['original_report_status'] = result.get('status')
            if result.get('status') != 'completed' or result.get('cleanup_errors'):
                report['cleanup_errors'].append('Original timestamp replay did not complete cleanly')
            result.update(internal_precision_changed=True, calibrated_retained_aggregator=True,
                requested_aggregator_quantized_dtype='w8a8', original_aggregator_ONNX_unchanged=True,
                legacy_precision_label_does_not_describe_aggregator=True,
                DINO_depth_and_SL4_unchanged=True, compiled_camera_internal_precision_changed=True,
                camera_adapter_source_sha256=args.camera_adapter_sha256,
                calibrated_entrypoint_receipt=str(receipt),
                quantized_manifest_sha256=args.quantized_manifest_sha256)
            result.update(calibrated_camera_head=True, requested_camera_quantized_dtype='w8a8',
                original_camera_ONNX_unchanged=True, legacy_precision_label_does_not_describe_camera=True,
                camera_compile_receipt_sha256=args.camera_compiled_sha256)
            result.setdefault('runtime_source_sha256', {})['private/rknn_calibrated_retained.py'] = args.adapter_sha256
            result['runtime_source_sha256']['private/rknn_calibrated_camera.py'] = args.camera_adapter_sha256
            write(result_path, result)
            report['annotated_report_sha256'] = sha(result_path)
        elif report['status'].startswith('completed_'):
            report['cleanup_errors'].append('Original timestamp replay report is missing')
        report['seconds'] = time.monotonic() - started
        if (report['cleanup_errors'] or not report['sources_unchanged'] or not report['installer_restored']
                or not report['camera_resources_closed']):
            report['status'] = 'failed_cleanup_or_identity'
        write(receipt, report)
        print(json.dumps({k: report.get(k) for k in ('status', 'installer_calls', 'cleanup_errors')},
                         allow_nan=False), flush=True)
    return 0 if report['status'].startswith('completed_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
