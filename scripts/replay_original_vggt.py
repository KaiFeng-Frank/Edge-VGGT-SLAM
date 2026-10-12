#!/usr/bin/env python3
"""RK3588S original VGGT + SL4, latest-frame TUM timestamp replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb-index', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-size', type=int, default=112)
    p.add_argument('--model-threads', type=int, choices=(1, 2, 4), default=4)
    p.add_argument('--model-cpus', nargs='+', type=int, choices=(4, 5, 6, 7),
                   help='Model CPU affinity; defaults to 6,7 for two threads and 4,5,6,7 otherwise')
    p.add_argument('--no-patch-cache', action='store_true')
    p.add_argument('--rknn-manifest', type=Path)
    p.add_argument('--rknn-manifest-sha256')
    p.add_argument('--rknn-paired-aggregator', action='store_true',
                   help='Use all 24 original frame/global pairs with both original outputs')
    p.add_argument('--rknn-retained-aggregator', action='store_true',
                   help='Group all original attention blocks at retained consumers, including target tokens')
    p.add_argument('--half-io-library', type=Path)
    p.add_argument('--half-io-library-sha256')
    p.add_argument('--rknn-camera-head', action='store_true')
    p.add_argument('--rknn-camera-split4', action='store_true',
                   help='Use the measured original four-way FC1 camera graph; quality remains unaccepted')
    p.add_argument('--rknn-dino-manifest', type=Path)
    p.add_argument('--rknn-dino-manifest-sha256')
    p.add_argument('--rknn-grouped-dino', action='store_true',
                   help='Execute all 24 original DINO layers as six four-layer graphs')
    p.add_argument('--rknn-depth-manifest', type=Path)
    p.add_argument('--rknn-depth-manifest-sha256')
    args = p.parse_args()
    cpus = (set(args.model_cpus) if args.model_cpus is not None else
            {6, 7} if args.model_threads == 2 else {4, 5, 6, 7})
    if args.model_cpus is not None and len(args.model_cpus) != len(cpus):
        p.error('--model-cpus must not contain duplicate CPUs')
    if len(cpus) < args.model_threads:
        p.error('--model-cpus must contain at least --model-threads CPUs')
    if args.rknn_retained_aggregator and (not args.rknn_paired_aggregator or args.half_io_library is None):
        p.error('--rknn-retained-aggregator requires --rknn-paired-aggregator and --half-io-library')
    if args.rknn_grouped_dino and (args.rknn_dino_manifest is None or args.half_io_library is None):
        p.error('--rknn-grouped-dino requires --rknn-dino-manifest and --half-io-library')
    if (args.half_io_library is None) != (args.half_io_library_sha256 is None):
        p.error('--half-io-library and --half-io-library-sha256 must be supplied together')
    if args.half_io_library is not None:
        if not args.rknn_paired_aggregator:
            p.error('--half-io-library requires --rknn-paired-aggregator')
        if not re.fullmatch('[0-9a-f]{64}', args.half_io_library_sha256):
            p.error('--half-io-library-sha256 must be a lowercase SHA256 hex digest')
    if args.rknn_paired_aggregator and args.rknn_manifest is None:
        p.error('--rknn-paired-aggregator requires --rknn-manifest')
    if args.rknn_camera_head and args.rknn_manifest is None:
        p.error('--rknn-camera-head requires --rknn-manifest')
    if args.rknn_camera_split4 and not args.rknn_camera_head:
        p.error('--rknn-camera-split4 requires --rknn-camera-head')
    if (args.rknn_manifest is None) != (args.rknn_manifest_sha256 is None):
        p.error('--rknn-manifest and --rknn-manifest-sha256 must be supplied together')
    if (args.rknn_dino_manifest is None) != (args.rknn_dino_manifest_sha256 is None):
        p.error('--rknn-dino-manifest and --rknn-dino-manifest-sha256 must be supplied together')
    if args.rknn_dino_manifest is not None:
        if not re.fullmatch('[0-9a-f]{64}', args.rknn_dino_manifest_sha256):
            p.error('--rknn-dino-manifest-sha256 must be a lowercase SHA256 hex digest')
        if args.image_size != 112:
            p.error('The fixed original RKNN DINO manifest requires --image-size 112')
    if (args.rknn_depth_manifest is None) != (args.rknn_depth_manifest_sha256 is None):
        p.error('--rknn-depth-manifest and --rknn-depth-manifest-sha256 must be supplied together')
    if args.rknn_depth_manifest is not None:
        if not re.fullmatch('[0-9a-f]{64}', args.rknn_depth_manifest_sha256):
            p.error('--rknn-depth-manifest-sha256 must be a lowercase SHA256 hex digest')
        if args.image_size != 112:
            p.error('The fixed original RKNN depth manifest requires --image-size 112')
    if args.rknn_manifest is not None:
        if not re.fullmatch('[0-9a-f]{64}', args.rknn_manifest_sha256):
            p.error('--rknn-manifest-sha256 must be a lowercase SHA256 hex digest')
        if args.image_size != 112:
            p.error('The fixed original RKNN aggregator manifest requires --image-size 112')
    compatible = next((x.read_bytes().replace(b'\0', b',').decode(errors='replace')
                       for x in (Path('/proc/device-tree/compatible'),
                                 Path('/sys/firmware/devicetree/base/compatible')) if x.is_file()), '')
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or 'rk3588' not in compatible:
        p.error('Actual execution is allowed on Linux RK3588S only')
    if args.image_size < 14 or args.image_size % 14:
        p.error('--image-size must be a positive multiple of 14')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', model='original_VGGT',
                  precision=('NPU_FP16_aggregator_camera_CPU_FP32_DINO_depth' if args.rknn_camera_head else
                             'NPU_FP16_aggregator_CPU_FP32_DINO_heads' if args.rknn_manifest is not None else 'CPU_FP32'),
                  max_loops=0, candidate_accepted=False, trajectory_quality_validated=False,
                  rknn_manifest=(str(args.rknn_manifest.resolve()) if args.rknn_manifest is not None else None),
                  rknn_manifest_sha256=args.rknn_manifest_sha256,
                  rknn_paired_aggregator=args.rknn_paired_aggregator,
                  rknn_camera_head=args.rknn_camera_head,
                  rknn_dino_manifest=(str(args.rknn_dino_manifest.resolve()) if args.rknn_dino_manifest else None),
                  rknn_dino_manifest_sha256=args.rknn_dino_manifest_sha256,
                  rknn_depth_manifest=(str(args.rknn_depth_manifest.resolve()) if args.rknn_depth_manifest else None),
                  rknn_depth_manifest_sha256=args.rknn_depth_manifest_sha256,
                  model_threads=args.model_threads,
                  image_size=args.image_size, patch_cache=not args.no_patch_cache, board=compatible,
                  rgb_index=str(args.rgb_index.resolve()), checkpoint=str(args.checkpoint.resolve()),
                  fresh_pose_count=0, updates=[], cleanup_errors=[], decoder_cpu_affinity=[0],
                  real_time_validated=False, short_replay_not_30min=True,
                  latency_scope='original software arrival through decode, queue, full model, SL4, dense chunk and poses/map atomic replacements',
                  physical_sensor_latency_validated=False)
    if args.rknn_dino_manifest is not None or args.rknn_depth_manifest is not None:
        report['precision'] = ('NPU_FP16' + ('_DINO' if args.rknn_dino_manifest else '') +
                               ('_aggregator' if args.rknn_manifest else '') +
                               ('_camera' if args.rknn_camera_head else '') +
                               ('_depth_logits' if args.rknn_depth_manifest else '') + '_remaining_CPU_FP32')
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['rknn_grouped_dino'] = args.rknn_grouped_dino
    report['rknn_retained_aggregator'] = args.rknn_retained_aggregator
    report['rknn_camera_split4'] = args.rknn_camera_split4
    report['runtime_source_sha256'] = {
        name: hashlib.sha256((BASE / 'src/vggt_slam' / name).read_bytes()).hexdigest()
        for name in ('patch_cache.py', 'rknn_aggregator_blocks.py', 'rknn_camera_head.py',
                     'rknn_dino_blocks.py', 'rknn_depth_head.py') +
                    (('rknn_aggregator_pairs.py',) if args.rknn_paired_aggregator else ())}
    if args.half_io_library is not None:
        half_io_sources = ('rknn_half_io_pairs.py', 'native/fp16_neon.c') + (
            ('rknn_half_io_dino.py',) if args.rknn_dino_manifest is not None else ())
        if args.rknn_grouped_dino:
            half_io_sources += ('rknn_grouped_dino.py',)
        if args.rknn_retained_aggregator:
            half_io_sources += ('rknn_retained_aggregator.py',)
        for name in half_io_sources:
            report['runtime_source_sha256'][name] = hashlib.sha256((BASE / 'src/vggt_slam' / name).read_bytes()).hexdigest()
        report.update(half_io_experiment=True, internal_precision_changed=False,
                      half_io_dino=args.rknn_dino_manifest is not None,
                      half_io_library=str(args.half_io_library.resolve()),
                      half_io_library_sha256=args.half_io_library_sha256,
                      half_io_source_sha256=report['runtime_source_sha256']['native/fp16_neon.c'])
        if args.rknn_retained_aggregator:
            report.update(internal_precision_changed=None,
                          compiled_boundary_rounding_equivalence_validated=False)
    if args.rknn_camera_split4:
        report.update(compiled_graph_changed=True, internal_precision_changed=None,
                      compiled_camera_rounding_equivalence_validated=False)
    replay = cache = depth_rknn = dino_rknn = camera_rknn = rknn = None
    hooks = []
    affinity = os.sched_getaffinity(0)
    setup = time.monotonic()
    try:
        if not (cpus | {0}) <= affinity:
            raise RuntimeError(f'Decoder CPU0 or selected model CPUs {sorted(cpus)} are unavailable')
        os.sched_setaffinity(0, cpus)
        sys.path[:0] = [str(BASE / 'src'), str(BASE / 'src/third_party/vggt')]
        import cv2
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        from vggt_slam.solver import Solver
        from vggt_slam.timestamp_replay import read_rgb_index, TimestampReplay
        from vggt_slam.edge_mapping import atomic_json, platform_sample
        cv2.setNumThreads(1)
        torch.set_num_threads(args.model_threads)
        torch.set_num_interop_threads(1)
        rows = read_rgb_index(args.rgb_index)
        model = load_model(args.checkpoint, device='cpu', image_size=args.image_size,
                           prune_unused_heads=True, offline=True)
        forward_clock = {}
        def before_forward(module, inputs):
            forward_clock['begin'] = time.monotonic()
        def after_forward(module, inputs, output):
            forward_clock['seconds'] = time.monotonic() - forward_clock['begin']
        hooks = [model.register_forward_pre_hook(before_forward), model.register_forward_hook(after_forward)]
        if args.rknn_manifest is not None:
            aggregator_options = dict(source_root=BASE / 'src')
            if args.half_io_library is not None:
                if args.rknn_retained_aggregator:
                    from vggt_slam.rknn_retained_aggregator import install_retained_aggregator as install_aggregator
                else:
                    from vggt_slam.rknn_half_io_pairs import install_half_io_pairs as install_aggregator
                aggregator_options.update(half_io_library=args.half_io_library,
                                           half_io_library_sha256=args.half_io_library_sha256)
            elif args.rknn_paired_aggregator:
                from vggt_slam.rknn_aggregator_pairs import install_rknn_aggregator_pairs as install_aggregator
            else:
                from vggt_slam.rknn_aggregator_blocks import install_rknn_aggregator_blocks as install_aggregator
            rknn = install_aggregator(model, args.rknn_manifest, args.rknn_manifest_sha256,
                                     **aggregator_options)
        if args.rknn_camera_head:
            from vggt_slam.rknn_camera_head import install_rknn_camera_head
            camera_rknn = (install_rknn_camera_head(model, split_fc1=True) if args.rknn_camera_split4
                           else install_rknn_camera_head(model))
        if args.rknn_dino_manifest is not None:
            dino_options = dict(source_root=BASE / 'src')
            if args.half_io_library is not None:
                if args.rknn_grouped_dino:
                    from vggt_slam.rknn_grouped_dino import install_grouped_dino as install_dino
                else:
                    from vggt_slam.rknn_half_io_dino import install_half_io_dino as install_dino
                dino_options.update(half_io_library=args.half_io_library,
                                    half_io_library_sha256=args.half_io_library_sha256)
            else:
                from vggt_slam.rknn_dino_blocks import install_rknn_dino_blocks as install_dino
            dino_rknn = install_dino(model, args.rknn_dino_manifest,
                                     args.rknn_dino_manifest_sha256, **dino_options)
        if args.rknn_depth_manifest is not None:
            from vggt_slam.rknn_depth_head import install_rknn_depth_head
            depth_rknn = install_rknn_depth_head(model, args.rknn_depth_manifest,
                                                args.rknn_depth_manifest_sha256, source_root=BASE / 'src')
        if not args.no_patch_cache:
            from vggt_slam.patch_cache import install_patch_cache
            cache = install_patch_cache(model)
        solver = Solver(init_conf_threshold=25., device='cpu', image_size=args.image_size,
                        max_loops=0, offline=True, enable_viewer=False)
        chunks = args.output / 'map_chunks'
        chunks.mkdir()
        manifest = dict(format='vggt_camera_local_dense_chunks_with_SL4_graph', units='model_units',
                        chunks=[], nodes={}, revision=0, metric_scale_validated=False, rectification=None)
        report.update(startup_seconds=time.monotonic()-setup, torch_version=torch.__version__, original_frames=len(rows),
                      cpu_affinity=sorted(os.sched_getaffinity(0)), platform_start=platform_sample())
        replay = TimestampReplay(rows, cpus={0})
        replay.start()
        anchor = None
        while True:
            item = replay.read()
            if item is None:
                if replay.exhausted:
                    break
                continue
            # Paths retain original identities; the unchanged PIL loader reads
            # these original pixels again inside the timed prediction call.
            item.pop('frame', None)
            if anchor is None:
                anchor = item
                report['seed_timestamp'] = item['timestamp']
                continue
            if item['timestamp'] <= anchor['timestamp']:
                raise RuntimeError('A new tail must have a strictly newer original timestamp')
            begin = time.monotonic()
            paths = [anchor['path'], item['path']]
            prediction = solver.run_predictions(paths, model, max_loops=0)
            predicted = time.monotonic()
            current = solver.current_working_submap
            if current.frame_ids != [anchor['timestamp'], item['timestamp']]:
                raise RuntimeError('TUM path timestamps differ from the original RGB index')
            solver.add_points(prediction)
            solver.graph.optimize()
            graph_error = float(solver.graph.graph.error(solver.graph.values))
            if not np.isfinite(graph_error):
                raise RuntimeError('SL4 graph error is nonfinite')
            optimized = time.monotonic()
            arrays, counts = {}, []
            for i in range(len(current.pointclouds)):
                mask = current.conf_masks[i] > current.conf_threshold
                arrays[f'points_{i}'] = np.asarray(current.pointclouds[i][mask], np.float32)
                arrays[f'colors_{i}'] = np.asarray(current.colors[i][mask], np.uint8)
                counts.append(len(arrays[f'points_{i}']))
            node = current.get_id()
            name = f'submap_{node:06d}.npz'
            np.savez(chunks / name, **arrays)
            manifest['chunks'].append(dict(path=f'map_chunks/{name}', submap_id=node,
                                           nodes=[node+i for i in range(len(counts))], point_counts=counts))
            for first, submap in sorted(solver.map.submaps.items()):
                for i in range(len(submap.pointclouds)):
                    H = np.asarray(solver.graph.get_homography(first+i), np.float64)
                    if H.shape != (4, 4) or not np.isfinite(H).all():
                        raise RuntimeError('SL4 graph homography is invalid')
                    manifest['nodes'][str(first+i)] = H.tolist()
            manifest['revision'] += 1
            temporary = args.output / 'poses.tum.tmp'
            solver.map.write_poses_to_file(str(temporary), solver.graph, kitti_format=False)
            temporary.replace(args.output / 'poses.tum')
            atomic_json(args.output / 'map.json', manifest)
            published = time.monotonic()
            row = dict(revision=manifest['revision'], timestamp=item['timestamp'], sequence=item['sequence'],
                       anchor_timestamp=anchor['timestamp'], captured_monotonic=item['captured_monotonic'],
                       published_monotonic=published, arrival_to_decode_seconds=item['decode_started_monotonic']-item['captured_monotonic'],
                       decode_seconds=item['ready_monotonic']-item['decode_started_monotonic'],
                       queue_seconds=begin-item['ready_monotonic'], prediction_seconds=predicted-begin,
                       full_model_seconds=forward_clock['seconds'], SL4_and_points_seconds=optimized-predicted,
                       publication_seconds=published-optimized, processing_seconds=published-begin,
                       input_to_pose_seconds=published-item['captured_monotonic'], graph_error=graph_error,
                       graph_nodes=int(solver.graph.values.size()), platform=platform_sample())
            report['updates'].append(row)
            report['fresh_pose_count'] += 1
            anchor = item
            print(json.dumps(row, allow_nan=False), flush=True)
            del prediction, arrays
        if not report['updates']:
            raise RuntimeError('Replay produced no complete original VGGT+SL4 pair')
        duration = report['updates'][-1]['published_monotonic'] - replay.start_monotonic
        report.update(status='completed', fresh_wall_seconds_including_EOF_drain=duration,
                      fresh_pose_hz=report['fresh_pose_count']/duration,
                      input_to_pose_p95_seconds=float(np.percentile([r['input_to_pose_seconds'] for r in report['updates']], 95)))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        # Stop arrivals, then restore cache before any model backend.
        for handle in (replay, cache, depth_rknn, dino_rknn, camera_rknn, rknn):
            if handle is not None:
                try:
                    handle.close()
                except BaseException as exc:
                    report['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
        for hook in hooks:
            hook.remove()
        if replay is not None:
            report['source'] = replay.metrics()
        if cache is not None:
            report['cache'] = cache.metrics()
        if camera_rknn is not None:
            report['camera_rknn'] = camera_rknn.metrics()
        if dino_rknn is not None:
            report['dino_rknn'] = dino_rknn.metrics()
        if depth_rknn is not None:
            report['depth_rknn'] = depth_rknn.metrics()
        if rknn is not None:
            report['rknn'] = rknn.metrics()
        try:
            os.sched_setaffinity(0, affinity)
        except OSError as exc:
            report['cleanup_errors'].append(f'affinity restore: {exc}')
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if report['cleanup_errors']:
            report['status'] = 'failed'
        temporary = args.output / 'results.json.tmp'
        temporary.write_text(json.dumps(report, allow_nan=False, indent=2))
        temporary.replace(args.output / 'results.json')
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
