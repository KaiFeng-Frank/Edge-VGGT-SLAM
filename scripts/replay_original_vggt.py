#!/usr/bin/env python3
"""RK3588S original VGGT FP32 + SL4, latest-frame TUM timestamp replay."""
import argparse
import json
import os
from pathlib import Path
import platform
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
    p.add_argument('--no-patch-cache', action='store_true')
    args = p.parse_args()
    compatible = next((x.read_bytes().replace(b'\0', b',').decode(errors='replace')
                       for x in (Path('/proc/device-tree/compatible'),
                                 Path('/sys/firmware/devicetree/base/compatible')) if x.is_file()), '')
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or 'rk3588' not in compatible:
        p.error('Actual execution is allowed on Linux RK3588S only')
    if args.image_size < 14 or args.image_size % 14:
        p.error('--image-size must be a positive multiple of 14')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', model='original_VGGT', precision='CPU_FP32', max_loops=0,
                  image_size=args.image_size, patch_cache=not args.no_patch_cache, board=compatible,
                  rgb_index=str(args.rgb_index.resolve()), checkpoint=str(args.checkpoint.resolve()),
                  fresh_pose_count=0, updates=[], cleanup_errors=[], decoder_cpu_affinity=[0],
                  real_time_validated=False, short_replay_not_30min=True,
                  latency_scope='original software arrival through decode, queue, full model, SL4, dense chunk and poses/map atomic replacements',
                  physical_sensor_latency_validated=False)
    replay = cache = None
    hooks = []
    affinity = os.sched_getaffinity(0)
    setup = time.monotonic()
    try:
        cpus = {4, 5, 6, 7}
        if not (cpus | {0}) <= affinity:
            raise RuntimeError('Decoder CPU0 or four A76 cores 4,5,6,7 are unavailable')
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
        torch.set_num_threads(4)
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
        for handle in (replay, cache):
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
