#!/usr/bin/env python3
"""Matched real TUM inputs through complete original VGGT and original SL4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--paired-aggregator', action='store_true')
    p.add_argument('--retained-aggregator', action='store_true',
                   help='Group all original attention blocks at retained consumers, including target tokens')
    p.add_argument('--save-aggregator-outputs', action='store_true',
                   help='Save retained layer tensors and target tokens after timed service for matched checks')
    p.add_argument('--half-io-library', type=Path)
    p.add_argument('--half-io-library-sha256')
    p.add_argument('--model-threads', type=int, choices=(1, 2, 4), default=4)
    p.add_argument('--model-cpus', nargs='+', type=int, choices=(4, 5, 6, 7),
                   help='Model CPU affinity; defaults to 6,7 for two threads and 4,5,6,7 otherwise')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dino-manifest', type=Path)
    p.add_argument('--dino-manifest-sha256')
    p.add_argument('--grouped-dino', action='store_true',
                   help='Execute all 24 original DINO layers as six four-layer graphs')
    p.add_argument('--depth-manifest', type=Path)
    p.add_argument('--depth-manifest-sha256')
    p.add_argument('--checkpoint-pageout', action='store_true',
                   help='Reclaim clean original checkpoint pages after each backend install; explicit experiment')
    p.add_argument('--camera-split4', action='store_true',
                   help='Use the measured original four-way FC1 camera graph in the complete DINO branch')
    p.add_argument('--reference-run', type=Path,
                   help='Reuse a completed matched CPU FP32 run; run only the new DINO branch')
    args = p.parse_args()
    cpus = (set(args.model_cpus) if args.model_cpus is not None else
            {6, 7} if args.model_threads == 2 else {4, 5, 6, 7})
    if args.model_cpus is not None and len(args.model_cpus) != len(cpus):
        p.error('--model-cpus must not contain duplicate CPUs')
    if len(cpus) < args.model_threads:
        p.error('--model-cpus must contain at least --model-threads CPUs')
    if args.retained_aggregator and (not args.paired_aggregator or args.half_io_library is None):
        p.error('--retained-aggregator requires --paired-aggregator and --half-io-library')
    if args.grouped_dino and (args.dino_manifest is None or args.half_io_library is None):
        p.error('--grouped-dino requires --dino-manifest and --half-io-library')
    if (args.half_io_library is None) != (args.half_io_library_sha256 is None):
        p.error('--half-io-library and --half-io-library-sha256 must be supplied together')
    if args.half_io_library is not None:
        if not args.paired_aggregator:
            p.error('--half-io-library requires --paired-aggregator')
        if not re.fullmatch('[0-9a-f]{64}', args.half_io_library_sha256):
            p.error('--half-io-library-sha256 must be a lowercase SHA256 hex digest')
    if (args.dino_manifest is None) != (args.dino_manifest_sha256 is None):
        p.error('Supply both DINO manifest and SHA256')
    if args.reference_run is not None and args.dino_manifest is None:
        p.error('--reference-run requires the DINO candidate')
    if args.camera_split4 and args.dino_manifest is None:
        p.error('--camera-split4 requires the complete DINO candidate')
    if (args.depth_manifest is None) != (args.depth_manifest_sha256 is None):
        p.error('Supply both depth manifest and SHA256')
    if args.depth_manifest is not None and args.dino_manifest is None:
        p.error('The complete depth comparison requires the DINO candidate')
    if args.checkpoint_pageout and not (args.retained_aggregator and args.grouped_dino
            and args.depth_manifest is not None and args.camera_split4
            and args.half_io_library is not None and args.reference_run is not None):
        p.error('--checkpoint-pageout requires the complete retained/grouped-DINO/depth/camera-split4/half-IO candidate and --reference-run')
    assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, cpus)
    cpu_affinity = sorted(os.sched_getaffinity(0))
    import numpy as np
    import torch
    from vggt_slam.runtime import load_model
    from vggt_slam.solver import Solver
    from vggt_slam.patch_cache import install_patch_cache
    from vggt_slam.timestamp_replay import read_rgb_index
    aggregator_options = {}
    if args.half_io_library is not None:
        if args.retained_aggregator:
            from vggt_slam.rknn_retained_aggregator import install_retained_aggregator as install_aggregator
        else:
            from vggt_slam.rknn_half_io_pairs import install_half_io_pairs as install_aggregator
        aggregator_options.update(half_io_library=args.half_io_library,
                                   half_io_library_sha256=args.half_io_library_sha256)
    elif args.paired_aggregator:
        from vggt_slam.rknn_aggregator_pairs import install_rknn_aggregator_pairs as install_aggregator
    else:
        from vggt_slam.rknn_aggregator_blocks import install_rknn_aggregator_blocks as install_aggregator
    from vggt_slam.rknn_camera_head import install_rknn_camera_head
    torch.set_num_threads(args.model_threads)
    torch.set_num_interop_threads(1)
    rows = read_rgb_index(BASE/'datasets/rgbd_dataset_freiburg1_xyz/rgb.txt')
    indices = list(range(32, 798, 48)) + [797]
    selected = [rows[i] for i in indices]
    report = dict(status='running', original_model_only=True, input_indices=indices,
                  input_timestamps=[r['timestamp'] for r in selected], branches={},
                  image_shape=[2,3,84,112], strict_rtol=1e-4, strict_atol=1e-5,
                  quality_accepted=False, realtime_validated=False,
                  paired_aggregator=args.paired_aggregator,
                  model_threads=args.model_threads,
                  cpu_affinity=cpu_affinity,
                  timing_scope='offline complete original model + SL4; not paced input latency',
                  manifest_sha256=args.manifest_sha256, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report['grouped_dino'] = args.grouped_dino
    report['retained_aggregator'] = args.retained_aggregator
    report['camera_split4'] = args.camera_split4
    report['checkpoint_pageout'] = args.checkpoint_pageout
    if args.camera_split4:
        report.update(compiled_graph_changed=True, compiled_camera_rounding_equivalence_validated=False)
    report['save_aggregator_outputs'] = args.save_aggregator_outputs
    report['runtime_source_sha256'] = {
        name: hashlib.sha256((BASE / 'src/vggt_slam' / name).read_bytes()).hexdigest()
        for name in ('patch_cache.py', 'rknn_aggregator_blocks.py', 'rknn_camera_head.py',
                     'rknn_dino_blocks.py', 'rknn_depth_head.py') +
                    (('rknn_aggregator_pairs.py',) if args.paired_aggregator else ())}
    if args.half_io_library is not None:
        half_io_sources = ('rknn_half_io_pairs.py', 'native/fp16_neon.c') + (
            ('rknn_half_io_dino.py',) if args.dino_manifest is not None else ())
        if args.grouped_dino:
            half_io_sources += ('rknn_grouped_dino.py',)
        if args.retained_aggregator:
            half_io_sources += ('rknn_retained_aggregator.py',)
        for name in half_io_sources:
            report['runtime_source_sha256'][name] = hashlib.sha256((BASE / 'src/vggt_slam' / name).read_bytes()).hexdigest()
        report.update(half_io_experiment=True, internal_precision_changed=False,
                      half_io_dino=args.dino_manifest is not None,
                      half_io_library=str(args.half_io_library.resolve()),
                      half_io_library_sha256=args.half_io_library_sha256,
                      half_io_source_sha256=report['runtime_source_sha256']['native/fp16_neon.c'])
        if args.retained_aggregator:
            report.update(internal_precision_changed=None,
                          compiled_boundary_rounding_equivalence_validated=False)
    if args.camera_split4:
        report['internal_precision_changed'] = None
    if args.checkpoint_pageout:
        report['runtime_source_sha256']['checkpoint_residency.py'] = hashlib.sha256(
            (BASE/'src/vggt_slam/checkpoint_residency.py').read_bytes()).hexdigest()
    branches = ['cpu_fp32', 'npu_fp16_aggregator_cpu_fp32_heads',
                'npu_fp16_aggregator_camera_cpu_fp32_DINO_depth']
    dino_branch = 'npu_fp16_DINO_aggregator_camera_cpu_fp32_depth'
    if args.depth_manifest is not None:
        dino_branch = 'npu_fp16_DINO_aggregator_camera_depth_logits_cpu_fp32_boundaries'
        report['depth_manifest_sha256'] = args.depth_manifest_sha256
    if args.paired_aggregator:
        dino_branch += '_paired_aggregator'
    if args.retained_aggregator:
        dino_branch += '_retained_groups'
    reference = args.output / 'cpu_fp32'
    if args.dino_manifest is not None:
        branches.append(dino_branch)
        report['dino_manifest_sha256'] = args.dino_manifest_sha256
    if args.reference_run is not None:
        reference_report_path = args.reference_run / 'result.json'
        reference_report = json.loads(reference_report_path.read_text())
        if (reference_report.get('status') != 'completed_not_quality_acceptance'
                or reference_report.get('input_indices') != indices
                or reference_report.get('input_timestamps') != report['input_timestamps']
                or reference_report.get('image_shape') != report['image_shape']
                or reference_report.get('branches', {}).get('cpu_fp32', {}).get('status') != 'completed'):
            raise ValueError('Reference must be the completed original CPU run on identical inputs')
        reference = args.reference_run / 'cpu_fp32'
        report['reused_cpu_reference'] = dict(path=str(args.reference_run),
            result_sha256=hashlib.sha256(reference_report_path.read_bytes()).hexdigest(),
            timing_comparison_interleaved=False,
            array_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(reference.glob('pair-*.npz'))})
        if set(report['reused_cpu_reference']['array_sha256']) != {f'pair-{i:02d}.npz' for i in range(1, len(selected))}:
            raise ValueError('Reference must retain all original complete output arrays')
        branches = [dino_branch]
    loaded_checkpoint_stat = None
    if args.checkpoint_pageout:
        from vggt_slam.checkpoint_residency import checkpoint_stat, memory_snapshot, pageout_checkpoint
        loaded_checkpoint_stat = checkpoint_stat(BASE/'weights/model.pt')
    model = load_model(BASE/'weights/model.pt', device='cpu', image_size=112,
                       prune_unused_heads=True, offline=True)
    if args.checkpoint_pageout and checkpoint_stat(BASE/'weights/model.pt') != loaded_checkpoint_stat:
        raise ValueError('Original checkpoint changed across verified load_model')
    phase_starts, phase_seconds, hooks, aggregator_outputs = {}, {}, [], {}
    def clock_hooks(name):
        def before(module, inputs):
            phase_starts[name] = time.monotonic()
        def after(module, inputs, outputs):
            phase_seconds[name] = time.monotonic()-phase_starts[name]
            if name == 'aggregator_including_DINO' and args.save_aggregator_outputs:
                aggregator_outputs['value'] = outputs
        return before, after
    for name, module in (('DINO',model.aggregator.patch_embed),
                         ('aggregator_including_DINO',model.aggregator),
                         ('camera_head',model.camera_head),('depth_head',model.depth_head)):
        before,after = clock_hooks(name)
        hooks.extend((module.register_forward_pre_hook(before),module.register_forward_hook(after)))
    def pageout_after_install(stage, result, folder):
        if not args.checkpoint_pageout:
            return
        receipt_path = folder/f'checkpoint-pageout-{stage}.json'
        item = dict(stage=stage, file=receipt_path.relative_to(args.output).as_posix(), status='calling')
        result.setdefault('checkpoint_pageout_stages', []).append(item)
        (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        try:
            receipt = pageout_checkpoint(model, BASE/'weights/model.pt', loaded_checkpoint_stat,
                receipt_path, source_root=BASE/'src', stage=stage)
            item.update(status=receipt['status'], sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
        except BaseException:
            item['status'] = 'failed'
            raise
    npu = camera_npu = dino_npu = depth_npu = cache = None
    try:
        for branch in branches:
            folder = args.output/branch
            folder.mkdir()
            result = dict(updates=[], status='running')
            report['branches'][branch] = result
            if branch != 'cpu_fp32':
                npu = install_aggregator(model, args.manifest, args.manifest_sha256, **aggregator_options)
                pageout_after_install('aggregator_installed', result, folder)
            if branch in ('npu_fp16_aggregator_camera_cpu_fp32_DINO_depth', dino_branch):
                camera_npu = (install_rknn_camera_head(model, split_fc1=True)
                              if args.camera_split4 and branch == dino_branch else install_rknn_camera_head(model))
                pageout_after_install('camera_installed', result, folder)
            if branch == dino_branch:
                dino_options = {}
                if args.half_io_library is not None:
                    if args.grouped_dino:
                        from vggt_slam.rknn_grouped_dino import install_grouped_dino as install_dino
                    else:
                        from vggt_slam.rknn_half_io_dino import install_half_io_dino as install_dino
                    dino_options.update(half_io_library=args.half_io_library,
                                        half_io_library_sha256=args.half_io_library_sha256)
                else:
                    from vggt_slam.rknn_dino_blocks import install_rknn_dino_blocks as install_dino
                dino_npu = install_dino(model, args.dino_manifest, args.dino_manifest_sha256,
                                        **dino_options)
                pageout_after_install('DINO_installed', result, folder)
                if args.depth_manifest is not None:
                    from vggt_slam.rknn_depth_head import install_rknn_depth_head
                    depth_npu = install_rknn_depth_head(model, args.depth_manifest, args.depth_manifest_sha256)
                    pageout_after_install('depth_installed', result, folder)
            cache = install_patch_cache(model)
            solver = Solver(25.,device='cpu',image_size=112,max_loops=0,offline=True,enable_viewer=False)
            for i in range(1,len(selected)):
                paths = [selected[i-1]['path'],selected[i]['path']]
                start = time.monotonic()
                with torch.inference_mode():
                    pred = solver.run_predictions(paths, model, max_loops=0)
                end = time.monotonic()
                solver.add_points(pred)
                solver.graph.optimize()
                pose = solver.map.get_latest_submap().get_all_poses_world(solver.graph)[-1]
                if not np.isfinite(pose).all():
                    raise ValueError('Nonfinite original SL4 pose')
                row = dict(pair=i, timestamp=selected[i]['timestamp'], prediction_seconds=end-start,
                           complete_service_seconds=time.monotonic()-start,
                           model_phase_seconds=dict(phase_seconds),
                           graph_error=float(solver.graph.graph.error(solver.graph.values)))
                if args.checkpoint_pageout:
                    row['checkpoint_pageout_post_inference'] = memory_snapshot(BASE/'weights/model.pt', loaded_checkpoint_stat)
                arrays = {k:np.array(pred[k],copy=True) for k in ('pose_enc','extrinsic','intrinsic','depth','depth_conf')}
                np.savez(folder/f'pair-{i:02d}.npz',**arrays)
                if args.save_aggregator_outputs:
                    values = aggregator_outputs.pop('value')
                    if (not isinstance(values, tuple) or len(values) != 4 or len(values[0]) != 24
                            or values[1] != 5 or values[3] is not None):
                        raise ValueError('Original Aggregator four-result contract differs')
                    retained = {4, 11, 17, 23}
                    if any(value is not None for j, value in enumerate(values[0]) if j not in retained):
                        raise ValueError('Unexpected retained original layer')
                    captured = {f'layer_{j:02d}': values[0][j] for j in sorted(retained)}
                    captured['target_tokens'] = values[2]
                    saved = {}
                    for key, value in captured.items():
                        shape = (2, 53, 1024) if key == 'target_tokens' else (1, 2, 53, 2048)
                        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                                or value.device.type != 'cpu' or tuple(value.shape) != shape
                                or not torch.isfinite(value).all()):
                            raise ValueError(f'Original Aggregator output differs: {key}')
                        saved[key] = value.detach().numpy().copy()
                    path = folder/f'aggregator-{i:02d}.npz'
                    np.savez(path, **saved)
                    row['aggregator_artifact'] = dict(file=path.name,
                        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        captured_after_timed_service=True)
                    del values, captured, saved
                if branch != 'cpu_fp32':
                    row['original_fp32_comparison'] = {}
                    with np.load(reference/f'pair-{i:02d}.npz') as ref:
                        for key, actual in arrays.items():
                            expected = ref[key]
                            if (actual.shape != expected.shape or not np.isfinite(actual).all()
                                    or not np.isfinite(expected).all()):
                                raise ValueError(f'Original reference shape/finite mismatch: {key}')
                            delta = actual.astype(np.float64)-expected.astype(np.float64)
                            row['original_fp32_comparison'][key] = dict(
                                finite=bool(np.isfinite(actual).all()),
                                relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(expected.astype(np.float64))),
                                max_abs=float(np.abs(delta).max()),
                                strict_pass=bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5)))
                result['updates'].append(row)
                print(json.dumps(dict(branch=branch,**row),allow_nan=False),flush=True)
                (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                del pred,arrays
            solver.map.write_poses_to_file(str(folder/'poses.tum'),solver.graph,kitti_format=False)
            cache.close()
            result.update(status='completed', cache=cache.metrics())
            cache=None
            if depth_npu is not None:
                depth_npu.close();result['depth_rknn']=depth_npu.metrics();depth_npu=None
            if dino_npu is not None:
                dino_npu.close();result['dino_rknn']=dino_npu.metrics();dino_npu=None
            if camera_npu is not None:
                camera_npu.close();result['camera_rknn']=camera_npu.metrics();camera_npu=None
            if npu is not None:
                npu.close();result['rknn']=npu.metrics();npu=None
        report['status']='completed_not_quality_acceptance'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        cleanup_errors = []
        for handle in (cache,depth_npu,dino_npu,camera_npu,npu):
            if handle is not None:
                try:
                    handle.close()
                except BaseException as exc:
                    cleanup_errors.append(f'{type(exc).__name__}: {exc}')
        for hook in hooks:
            try:
                hook.remove()
            except BaseException as exc:
                cleanup_errors.append(f'{type(exc).__name__}: {exc}')
        report['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            report['status'] = 'failed'
        report['peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if args.checkpoint_pageout:
            try:
                report['checkpoint_pageout_post_cleanup_memory'] = memory_snapshot(BASE/'weights/model.pt', loaded_checkpoint_stat)
                report['post_cleanup_RSS_MiB'] = report['checkpoint_pageout_post_cleanup_memory']['status_kB']['VmRSS']/1024
            except BaseException as exc:
                cleanup_errors.append(f'pageout final snapshot: {type(exc).__name__}: {exc}')
                report['status'] = 'failed'
        (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if cleanup_errors:
            raise RuntimeError('Backend cleanup failed: ' + '; '.join(cleanup_errors))


if __name__ == '__main__':
    main()
