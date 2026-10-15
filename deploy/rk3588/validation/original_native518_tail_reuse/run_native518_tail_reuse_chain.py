#!/usr/bin/env python3
"""Exact previous-tail DINO reuse through two fresh original native518 SL4 pairs.

Board-only diagnostic: no timestamp pacing, ATE/RPE or realtime acceptance.
The original Solver, scale equations, factors and LM optimizer are unchanged.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2'); SRC = BASE/'src'; E = BASE/'experiments'
S100 = E/'vggt-v100-source-r1'
PAIR_CHAIN_SHA = '966900291273b6f61af151307f2713ede5524e132847bd1b61c372e9ccdb523c'
REFERENCE_SHA = '4af7b9c4cbd38dbfd54c14b1308dd6035da25941f98f858a5ad13b781c3c8af6'
NATIVE_SHA = '3cdfa55556896b0f5356f710c65069b8b5e53f37a0f8cb6f550439f6dfb06730'
CONFIG_SHA = '7e27bf329bfb37afa3b64f55969fc612df64455f7d6c8c11ac654faef709fd8b'
SOURCE_PINS = {
    'vggt_slam/model_execution.py': 'dc5365fd3e823385261b45be2b4b00085c9fb30afbc40825c6c55af0b0253e87',
    'vggt_slam/solver.py': '33b5b4d816cd08f0cb9857f89c9e3fc770ad0c2ab3c9dc938864f00cefe86905',
    'vggt_slam/graph.py': 'dda8fb7c947c3c39baa7d190e2942cd4b9f26fe1557d7605fdc8651f27a5ab48',
    'vggt_slam/submap.py': 'fddccafee4e9f32452ce044b80669bd48f5fe5f331a980986a6091e7fce92e63',
    'vggt_slam/map.py': '2d7b374101248519d921e16e272063ff6f7e001c7af0073a7a7c13c14ac79143',
    'vggt_slam/slam_utils.py': 'aeba669bcb263f24e6bd93e5a176bc028e2022770499bd5ea1c3eeeebbbd5008',
    'vggt_slam/scale_solver.py': 'dc36b11d4d9953222a3fc2f3aef274acc9e9cc6dc61cb22d86b1e6721d9bd691',
    'vggt_slam/frame_overlap.py': 'bbb78f549a34b58eaf27556e6dda703468cc0518de4b936f7f886882d151afb8',
    'vggt_slam/runtime.py': 'f36bb5b87fbdbf03984ea3e1466e6ee27865a540ae3e5abd4ee07a45538e23f5',
    'vggt_slam/checkpoint_identity.py': '9fc0b927f6a68ac6ca55194616104febb1c488370593d4be60e8dba5bc336461',
    'vggt_slam/timestamp_replay.py': '14b168028a6f7722f4bea036285b4f9cd8119fed59230b7e3e07c4b7af07addc',
}
INDICES = [32, 40, 48]
BASENAMES = ['1305031103.243216.png', '1305031103.511333.png', '1305031103.775342.png']
ALIASES = {'pose_enc':'pose_enc', 'extrinsic':'extrinsics', 'intrinsic':'intrinsics', 'depth':'depth', 'depth_conf':'depth_conf'}
SUCCESS = 'completed_exact_tail_DINO_reuse_SL4_chain_not_quality_or_realtime_accepted'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4 << 20), b''): h.update(data)
    return h.hexdigest()


def atomic(path, row):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(row, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executor-source', type=Path, default=Path(__file__).with_name('v110_native518_executor.py'))
    p.add_argument('--executor-sha256', required=True)
    p.add_argument('--core-mask', type=int, choices=(1,7), default=1)
    p.add_argument('--config', type=Path, default=S100/'full-native-sequential-config-r1.json')
    p.add_argument('--frozen-source-dir', type=Path, default=S100)
    p.add_argument('--rgb-index', type=Path, default=BASE/'datasets/rgbd_dataset_freiburg1_xyz/rgb.txt')
    p.add_argument('--baseline-reference', type=Path, default=Path('/dev/shm/edge-vggt-v100-native/full-model-sequential-core1-r1/result.json'))
    p.add_argument('--pair-chain-reference', type=Path, default=Path('/dev/shm/edge-vggt-v109-native/pair-chain-r1/result.json'))
    p.add_argument('--native-reference', type=Path, default=E/'vggt-v92-native-complete-r1/capture.json')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--timeout-seconds', type=float, default=170.)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588/S board only')
    require(0 < a.timeout_seconds <= 180, 'Existing external bounded180s controller required')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+a.timeout_seconds
    env, affinity, oldpath = dict(os.environ), os.sched_getaffinity(0), sys.path[:]
    handlers, pins = {}, {}; executor = solver = None
    r = dict(schema='v110_original_native518_tail_DINO_reuse_SL4_chain', status='running', script_sha256=sha(__file__),
        executor_sha256=a.executor_sha256, requested_core_mask=a.core_mask, original_model_only=True, max_loops=0, image_size=518,
        prepared_image_shape=[2,3,392,518], source_indices=INDICES, pairs=[], cleanup_errors=[],
        DINO_layers_per_frame=24, aggregator_blocks_per_pair=48, camera_refinements_per_pair=4, depth_frames_per_pair=2,
        traditional_pose_frontend_executed=False, timestamp_paced=False, seed_counted_as_fresh_pose=False,
        trajectory_quality_realtime_map_cadence_30min_or_8GB_accepted=False,
        timing_scope='Original file read/preprocessing, complete native model and pose codec, five prediction saves before backend mutation, original points/scale/factors/LM and pose-file publication. Both-pair comparisons to the uncached v109 outputs are included. Final whole-map PLY/homography/metadata export is recorded separately.',
        source_loading_and_checkpoint_startup_outside_pair_timing=True)
    atomic(out/'result.json', r)
    def pin(path, digest):
        path = Path(path).resolve(strict=True); require(sha(path) == digest, 'Pinned source/artifact changed: '+str(path))
        pins[path] = digest; return path
    def progress(stage, **fields):
        require(time.monotonic() < deadline, 'Diagnostic deadline reached at '+stage)
        print(json.dumps(dict(stage=stage, **fields)), flush=True)
    try:
        os.sched_setaffinity(0, {4,5,6,7}); os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        for number in (signal.SIGINT, signal.SIGTERM):
            def stop(n, frame): raise InterruptedError('External bounded controller stopped native SL4 diagnostic')
            handlers[number] = signal.signal(number, stop)
        pin(__file__, r['script_sha256']); source = pin(a.executor_source, a.executor_sha256); pin(a.config, CONFIG_SHA)
        for name, digest in SOURCE_PINS.items(): pin(SRC/name, digest)
        baseline_path = pin(a.baseline_reference, REFERENCE_SHA); old = json.loads(baseline_path.read_text())
        native = json.loads(pin(a.native_reference, NATIVE_SHA).read_text())
        pair_baseline_path = pin(a.pair_chain_reference, PAIR_CHAIN_SHA)
        pair_baseline = json.loads(pair_baseline_path.read_text())
        require(pair_baseline['status'] == 'completed_original_native518_two_pair_SL4_chain_not_quality_or_realtime_accepted'
            and pair_baseline['cleanup_errors'] == [] and pair_baseline['source_indices'] == INDICES
            and pair_baseline['executor_resources']['resources_balanced'], 'Clean same-input uncached v109 chain required')
        require(old['status'] == 'completed_actual_original_native_model_path_not_quality_or_realtime_accepted'
            and old['resources_balanced'] and old['complete_expected_execution'] and old['cleanup_errors'] == []
            and old['checkpoint'] == native['checkpoint'], 'Clean frozen full-model reference required')
        sys.path[:0] = [str(SRC), str(SRC/'third_party/vggt')]
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from vggt_slam.timestamp_replay import read_rgb_index
        from vggt_slam.solver import Solver
        records = read_rgb_index(a.rgb_index); pins[a.rgb_index.resolve()] = sha(a.rgb_index)
        require(len(records) > 48, 'Real TUM source index is too short')
        selected = [records[i] for i in INDICES]
        require([Path(v['path']).name for v in selected] == BASENAMES, 'Exact three native TUM frames required')
        require([v['timestamp'] for v in selected] == sorted({v['timestamp'] for v in selected}), 'Three strictly fresh source timestamps required')
        source_rows = []
        for index, record in zip(INDICES, selected):
            path = Path(record['path']); digest = sha(path); pins[path] = digest
            source_rows.append(dict(index=index, **record, bytes=path.stat().st_size, sha256=digest))
        require(len({v['sha256'] for v in source_rows}) == 3, 'New source file is identical to an earlier frame')
        for i in range(2):
            row = native['source_images'][i]
            require(Path(row['file']).resolve() == Path(selected[i]['path']) and row['sha256'] == source_rows[i]['sha256'], 'Initial two images differ from original native reference')
        references = {}
        for name, alias in ALIASES.items():
            row = old['calls'][0]['artifacts'][alias]; rel = Path(row['file'])
            require(not rel.is_absolute() and '..' not in rel.parts, 'Reference-relative output required')
            path = pin(baseline_path.parent/rel, row['sha256']); value = np.load(path, allow_pickle=False)
            require(value.dtype == np.float32 and list(value.shape) == row['shape'] and value.shape[0] == 1
                and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == row['raw_sha256'], 'Original complete reference array differs')
            references[name] = value[0]
        pair_references = []
        for item in pair_baseline['pairs']:
            output_refs = {}
            for name in ALIASES:
                row = item['outputs'][name]; rel = Path(row['file'])
                require(not rel.is_absolute() and '..' not in rel.parts, 'Pair-reference-relative output required')
                path = pin(pair_baseline_path.parent/rel, row['sha256']); value = np.load(path, allow_pickle=False)
                require(value.dtype == np.float32 and list(value.shape) == row['shape'] and np.isfinite(value).all()
                    and hashlib.sha256(value.tobytes()).hexdigest() == row['raw_sha256'], 'Uncached v109 output array differs')
                output_refs[name] = value
            pair_references.append(output_refs)
        require(len(pair_references) == 2 and pair_baseline['source_images'] == source_rows, 'Matched real source files required')
        r.update(pair_chain_reference_sha256=PAIR_CHAIN_SHA, matched_uncached_reference=True)
        r.update(source_images=source_rows, checkpoint=native['checkpoint'], reference_receipt_sha256=REFERENCE_SHA,
            original_backend_source_sha256=SOURCE_PINS, rgb_index_sha256=sha(a.rgb_index))
        spec = importlib.util.spec_from_file_location('_v110_original_native_executor', source)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
        progress('native_executor_initializing'); tick = time.monotonic()
        executor = module.Native518Executor(config=a.config, config_sha256=CONFIG_SHA, frozen_source_dir=a.frozen_source_dir,
            capture_intermediates=False, source_root=SRC, expected_npu_hz=1000000000,
            reuse_previous_tail=True, core_mask=a.core_mask)
        solver = Solver(init_conf_threshold=25., device='cpu', image_size=518, max_loops=0, offline=True, enable_viewer=False)
        r['startup_seconds'] = time.monotonic()-started; r['executor_init_seconds'] = time.monotonic()-tick
        atomic(out/'result.json', r); progress('native_executor_ready')
        previous_tail_raw = None; seen_pixels = set()
        def save(name, value):
            value = np.ascontiguousarray(value); path = out/(name+'.npy')
            with path.open('xb') as f: np.save(f, value, allow_pickle=False)
            return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), shape=list(value.shape), dtype=str(value.dtype),
                raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(), finite=bool(np.isfinite(value).all()))
        for pair_index in range(2):
            pair = selected[pair_index:pair_index+2]; paths = [v['path'] for v in pair]; stamps = [v['timestamp'] for v in pair]
            progress('fresh_pair_start', index=pair_index, timestamps=stamps); begin = time.monotonic()
            prediction = solver.run_predictions(paths, executor, max_loops=0); predicted = time.monotonic()
            pixels = prediction['images']; require(pixels.shape == (2,3,392,518), 'Reject112 or changed native pixel grid')
            raw = [hashlib.sha256(v.tobytes()).hexdigest() for v in pixels]
            require(raw[0] != raw[1] and (previous_tail_raw is None or raw[0] == previous_tail_raw)
                and raw[1] not in seen_pixels, 'Pair must share the exact previous image and introduce new pixels')
            seen_pixels.update(raw); previous_tail_raw = raw[1]
            saved_outputs = {name:save('pair'+str(pair_index)+'-'+name,prediction[name]) for name in ALIASES}
            parity = {name:bool(np.array_equal(prediction[name], ref) and prediction[name].tobytes() == ref.tobytes()) for name, ref in references.items()} if pair_index == 0 else None
            exact = {name:bool(np.array_equal(prediction[name], ref) and prediction[name].tobytes() == ref.tobytes())
                for name, ref in pair_references[pair_index].items()}
            r.setdefault('prediction_checks_before_SL4_commit', []).append(dict(index=pair_index,
                prepared_frame_raw_sha256=raw, first_pair_v100_byte_equal=parity,
                same_pair_uncached_v109_byte_equal=exact, outputs=saved_outputs))
            atomic(out/'result.json', r)
            require(parity is None or all(parity.values()), 'Initial complete five outputs differ from frozen v100')
            require(all(exact.values()), 'Complete five outputs differ from uncached same-input v109')
            current = solver.current_working_submap
            require(current.get_frame_ids() == stamps, 'Original Submap timestamps differ from source identities')
            backend_tick = time.monotonic()
            solver.add_points(prediction); solver.graph.optimize(); optimized = time.monotonic()
            nodes, factors = int(solver.graph.values.size()), int(solver.graph.graph.size())
            error = float(solver.graph.graph.error(solver.graph.values))
            require(nodes == factors == 2*(pair_index+1) and math.isfinite(error), 'Original no-loop SL4 chain counts or residual differ')
            point_rows = []
            for i in range(2):
                require(np.isfinite(current.pointclouds[i]).all(), 'Dense original point map is nonfinite')
                mask = current.conf_masks[i] > current.conf_threshold
                point_rows.append(int(mask.sum()))
            temporary = out/'poses.tum.tmp'; solver.map.write_poses_to_file(str(temporary), solver.graph, kitti_format=False); temporary.replace(out/'poses.tum')
            lines = (out/'poses.tum').read_text().splitlines(); trajectory = np.asarray([[float(v) for v in line.split()] for line in lines])
            require(trajectory.shape == (pair_index+2,8) and np.isfinite(trajectory).all()
                and np.allclose(trajectory[:,0], [v['timestamp'] for v in selected[:pair_index+2]], rtol=0, atol=1e-7), 'Original deduplicated TUM trajectory differs')
            published = time.monotonic()
            item = dict(index=pair_index, source_indices=INDICES[pair_index:pair_index+2], timestamps=stamps, image_paths=paths,
                prepared_image_shape=list(pixels.shape), prepared_frame_raw_sha256=raw, first_pair_v100_byte_equal=parity, same_pair_uncached_v109_byte_equal=exact,
                prediction_seconds=predicted-begin, identity_parity_and_prediction_save_seconds=backend_tick-predicted,
                points_scale_factors_LM_seconds=optimized-backend_tick, publication_seconds=published-optimized,
                image_read_to_original_pose_publication_seconds=published-begin, original_model_timings=executor.last_timings,
                executor_call=copy.deepcopy(executor.last_call),
                graph_nodes=nodes, graph_factors=factors, graph_error=error, map_submaps=solver.map.get_num_submaps(),
                current_filtered_points=point_rows, published_unique_timestamps=pair_index+2,
                outputs=saved_outputs)
            r['pairs'].append(item); atomic(out/'result.json', r)
            progress('fresh_pair_committed', index=pair_index, seconds=item['image_read_to_original_pose_publication_seconds'], graph_nodes=nodes, graph_error=error)
        progress('original_whole_map_export'); tick = time.monotonic()
        homographies = solver.map.get_all_homographies(solver.graph)
        require(homographies.shape == (4,4,4) and np.isfinite(homographies).all(), 'Four original SL4 homographies required')
        h_artifact = save('SL4-homographies', homographies)
        solver.map.write_points_to_file(solver.graph, str(out/'map.ply'))
        metadata = dict(format='original_SL4_map_metadata', units='original_model_gauge', metric_scale_validated=False,
            max_loops=0, submaps=2, graph_nodes=4, graph_factors=4, unique_timestamps=[v['timestamp'] for v in selected],
            nodes=[dict(submap_id=s.get_id(), node_ids=[s.get_id()+i for i in range(len(s.poses))],
                frame_timestamps=s.get_frame_ids(), filtered_point_counts=[int((s.conf_masks[i]>s.conf_threshold).sum()) for i in range(2)])
                for s in solver.map.ordered_submaps_by_key()], homographies=h_artifact,
            original_PLY_writer='GraphMap.write_points_to_file; original Submap.get_points_in_world_frame')
        atomic(out/'map.json', metadata)
        exported = time.monotonic()
        map_parity = {}
        for field, filename in (('map_PLY_artifact','map.ply'), ('poses_artifact','poses.tum')):
            row = pair_baseline[field]; reference = pin(pair_baseline_path.parent/row['file'],row['sha256'])
            map_parity[filename] = sha(out/filename) == row['sha256']
        ref_meta_row = pair_baseline['map_manifest_artifact']
        ref_meta = json.loads(pin(pair_baseline_path.parent/ref_meta_row['file'], ref_meta_row['sha256']).read_text())
        ref_h = ref_meta['homographies']
        reference = pin(pair_baseline_path.parent/ref_h['file'],ref_h['sha256'])
        map_parity['SL4-homographies.npy'] = sha(out/'SL4-homographies.npy') == ref_h['sha256']
        r['same_chain_uncached_v109_byte_equal'] = map_parity
        require(all(map_parity.values()), 'Original SL4/map changed despite equal model predictions')
        r.update(status=SUCCESS, fresh_tail_observations_committed=2, published_unique_timestamps=3,
            final_original_map_export_seconds=exported-tick,
            final_map_reference_verification_seconds=time.monotonic()-exported,
            map_PLY_artifact=dict(file='map.ply', bytes=(out/'map.ply').stat().st_size, sha256=sha(out/'map.ply')),
            poses_artifact=dict(file='poses.tum', bytes=(out/'poses.tum').stat().st_size, sha256=sha(out/'poses.tum')),
            map_manifest_artifact=dict(file='map.json', bytes=(out/'map.json').stat().st_size, sha256=sha(out/'map.json')))
    except BaseException as exc:
        executor = getattr(exc, 'native_executor', executor)
        r.update(status='failed', error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        if executor is not None:
            try: executor.close()
            except BaseException as exc: r['cleanup_errors'].append('Executor close: '+repr(exc))
            try:
                z = executor.metrics(); r['executor_resources'] = z
                if r['status'] == SUCCESS:
                    require(z['SDK_totals'] == dict(contexts_created=20, contexts_destroyed=20, runs_completed=32,
                        output_buffers_acquired=42, output_buffers_released=42) and z['closed'] and z['live_contexts'] == 0
                        and z['live_output_buffers'] == 0 and z['cleanup_errors'] == [], 'Complete20contexts/32runs/42owned outputs required')
            except BaseException as exc: r['cleanup_errors'].append('Executor resources: '+repr(exc))
        for number, handler in handlers.items(): signal.signal(number, handler)
        try: os.sched_setaffinity(0, affinity)
        except BaseException as exc: r['cleanup_errors'].append('Affinity restore: '+repr(exc))
        os.environ.clear(); os.environ.update(env); sys.path[:] = oldpath
        for path, digest in pins.items():
            try: require(sha(path) == digest, 'Pinned source/input changed: '+str(path))
            except BaseException as exc: r['cleanup_errors'].append(str(exc))
        r.update(source_and_inputs_unchanged=not r['cleanup_errors'], source_and_input_sha256={str(p):h for p,h in pins.items()},
            environment_restored=dict(os.environ)==env, affinity_restored=os.sched_getaffinity(0)==affinity, sys_path_restored=sys.path==oldpath,
            total_wall_seconds=time.monotonic()-started, peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if r['cleanup_errors'] or not r['environment_restored'] or not r['affinity_restored']: r['status'] = 'failed_cleanup'
        atomic(out/'result.json', r); print(json.dumps({k:r.get(k) for k in ('status','error','cleanup_errors')}), flush=True)
    return 0 if r['status'] == SUCCESS else 1


if __name__ == '__main__': raise SystemExit(main())
