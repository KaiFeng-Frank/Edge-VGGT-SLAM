#!/usr/bin/env python3
"""Independent JSON/NumPy/PNG/PLY audit only; never imports NN/ORT/SDK/GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = 'd645622ad58e4a7f291347b902ee5d55e16e12c0252e20f99c772cb32ba85f72'
EXECUTOR = 'e153a5f71ef0b10119207b2e43589b628a2e85ab1abeaeb72d2df2a7dd6fb638'
BASELINE = '966900291273b6f61af151307f2713ede5524e132847bd1b61c372e9ccdb523c'
ROUTE_PINS = {
    1: ('547ed32b951f1eeae40492cc01df15051d2088e4f588f33c5dc4a19b03f8a09a', '2360004d848ddf633fdb36bacb9e2798dab53f15224addee50d9e4899ae5e90e'),
    7: ('893d0e788a3550a680e2499921f89442a09c9698f0ed061aaa43821a499b081c', 'a43eabd07a86243ea50e66f57d1fa2ba8b9aa7129fe599202e7959e3057e8ccb'),
}
SHAPES = {'pose_enc': [2,9], 'extrinsic': [2,3,4], 'intrinsic': [2,3,3], 'depth': [2,392,518,1], 'depth_conf': [2,392,518]}
COUNTERS = ('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir', type=Path, default=ROOT/'uncached-v109-reference')
    ap.add_argument('--source-images', type=Path, default=ROOT/'source-images')
    ap.add_argument('--output', type=Path, default=ROOT/'saved-tail-reuse-audit.json')
    args = ap.parse_args()
    pins, arrays, routes = {}, [], {}
    oldenv = dict(os.environ)
    report = dict(schema='v110_independent_saved_tail_DINO_reuse_SL4_chain_audit', status='failed',
        auditor_sha256=sha(__file__), saved_data_only=True, model_ORT_SDK_GPU_board_execution=False,
        trajectory_quality_realtime_map_cadence_30min_or_8GB_accepted=False)
    try:
        os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        import numpy as np
        def pin(path, digest):
            path = Path(path).resolve(strict=True)
            require(sha(path) == digest, 'File SHA differs: '+str(path)); pins[path] = digest
            return path
        def artifact(directory, row):
            rel = Path(row['file'])
            require(not rel.is_absolute() and '..' not in rel.parts, 'Relative artifact required')
            path = pin(directory/rel, row['sha256'])
            require(path.is_relative_to(directory.resolve()) and path.stat().st_size == row['bytes'], 'Artifact size/path differs')
            return path
        def array(directory, row, dtype, shape):
            path = artifact(directory, row); value = np.load(path, allow_pickle=False)
            raw = hashlib.sha256(value.tobytes()).hexdigest()
            require(value.dtype == dtype and list(value.shape) == row['shape'] == shape and value.flags.c_contiguous
                and np.isfinite(value).all() and raw == row['raw_sha256'], 'Array shape/dtype/raw/finite differs')
            arrays.append(dict(file=str(path), bytes=path.stat().st_size, sha256=row['sha256'], raw_sha256=raw, shape=shape, finite=True))
            return value
        pin(ROOT/'run_native518_tail_reuse_chain.py', RUNNER); pin(ROOT/'v110_native518_executor.py', EXECUTOR)
        basepath = pin(args.baseline_dir/'result.json', BASELINE); base = json.loads(basepath.read_text())
        require(base['status'] == 'completed_original_native518_two_pair_SL4_chain_not_quality_or_realtime_accepted'
            and base['cleanup_errors'] == [] and base['prepared_image_shape'] == [2,3,392,518], 'Clean original uncached chain required')
        reference = [{name: array(args.baseline_dir, p['outputs'][name], np.float32, shape) for name, shape in SHAPES.items()} for p in base['pairs']]
        bm_path = artifact(args.baseline_dir, base['map_manifest_artifact']); bm = json.loads(bm_path.read_text())
        base_h = array(args.baseline_dir, bm['homographies'], np.float64, [4,4,4])
        for row in base['source_images']:
            path = pin(args.source_images/Path(row['path']).name, row['sha256'])
            require(path.stat().st_size == row['bytes'], 'Original source PNG size differs')
        for mask, (result_sha, control_sha) in ROUTE_PINS.items():
            directory = ROOT/f'tail-reuse-core{mask}-r1'
            rpath = pin(directory/'result.json', result_sha); result = json.loads(rpath.read_text())
            cpath = pin(ROOT/f'control-core{mask}-r1/control.json', control_sha); control = json.loads(cpath.read_text())
            require(result['schema'] == 'v110_original_native518_tail_DINO_reuse_SL4_chain'
                and result['status'] == 'completed_exact_tail_DINO_reuse_SL4_chain_not_quality_or_realtime_accepted'
                and result['script_sha256'] == RUNNER and result['executor_sha256'] == EXECUTOR
                and result['requested_core_mask'] == mask and result['pair_chain_reference_sha256'] == BASELINE
                and result['source_images'] == base['source_images'] and result['checkpoint'] == base['checkpoint']
                and result['prepared_image_shape'] == [2,3,392,518] and result['max_loops'] == 0
                and result['cleanup_errors'] == [] and all(result[k] for k in ('source_and_inputs_unchanged','environment_restored','affinity_restored','sys_path_restored'))
                and len(result['pairs']) == 2, 'Pinned complete original native route required')
            totals, per_pair = dict.fromkeys(COUNTERS,0), []
            for i, p in enumerate(result['pairs']):
                original = base['pairs'][i]; call = p['executor_call']; raw = p['prepared_frame_raw_sha256']
                require(p['index'] == i and p['source_indices'] == original['source_indices'] and p['timestamps'] == original['timestamps']
                    and raw == original['prepared_frame_raw_sha256'] == call['input_frame_raw_sha256']
                    and call['input_raw_sha256'] == original['executor_call']['input_raw_sha256']
                    and call['status'] == 'completed' and call['tail_cache_hit'] == bool(i)
                    and call['encoder_frame_indices_computed'] == ([1] if i else [0,1])
                    and call['actual_fresh_caller_pixels'] and not call['fixed_reference_pixel_substitution']
                    and not call['intermediate_reference_substitution'] and not call['prediction_or_aggregator_cache_used'], 'Fresh input/cache boundary differs')
                require(call['reused_frame_raw_sha256'] == (raw[0] if i else None), 'Wrong reused image identity')
                values = {name: array(directory,p['outputs'][name],np.float32,shape) for name, shape in SHAPES.items()}
                same = {name: bool(np.array_equal(v,reference[i][name]) and v.tobytes() == reference[i][name].tobytes()) for name,v in values.items()}
                require(all(same.values()), 'Prediction bytes differ from uncached original v109')
                for name, value in values.items():
                    require(call['final_outputs'][name]['raw_sha256'] == hashlib.sha256(value.tobytes()).hexdigest(), 'Executor raw output differs')
                resources = call['resources']; owners = [resources[f'dino{j}'] for j in range(4)] + resources['aggregator']['per_group_closed_contexts'] + resources['heads']['contexts']
                ledger = {key: sum(v[key] for v in owners) for key in COUNTERS}
                expected = dict(zip(COUNTERS,(10,10,14 if i else 18,19 if i else 23,19 if i else 23)))
                require(len(owners) == 10 and ledger == expected == call['SDK_totals']
                    and all(v['closed'] and not v['failed'] and not v['live_contexts'] and not v['live_output_buffers']
                        and v['cleanup_errors'] == [] and v['core_mask'] == mask and v['init_flags'] == 0 for v in owners), 'Actual SDK owner ledger differs')
                require(resources['aggregator']['completed_original_blocks'] == 48 and resources['heads']['completed_refinements'] == 4
                    and resources['heads']['completed_depth_frames'] == 2 and call['resources_balanced'] and call['complete_expected_execution'], 'Original layers/heads incomplete')
                for key in COUNTERS: totals[key] += ledger[key]
                cache = call['cache_after_call']
                require(cache['cache_entries'] == 1 and cache['cache_bytes'] == 6700608 and cache['lookups'] == i+1
                    and cache['hits'] == i and cache['misses'] == 1 and cache['commits'] == i+1
                    and cache['new_tail_always_computed'] and not cache['aggregator_head_pose_or_prediction_reuse'], 'Single exact-tail cache evidence differs')
                timing = ('prediction_seconds','identity_parity_and_prediction_save_seconds','points_scale_factors_LM_seconds','publication_seconds')
                require(all(math.isfinite(p[k]) and p[k] >= 0 for k in timing)
                    and math.isclose(sum(p[k] for k in timing), p['image_read_to_original_pose_publication_seconds'], abs_tol=1e-9)
                    and p['prediction_seconds'] >= call['complete_predict_boundary_seconds'] >= call['complete_image_to_heads_seconds'], 'Complete boundary timing differs')
                require(p['graph_nodes'] == p['graph_factors'] == 2*(i+1) and p['map_submaps'] == i+1 and p['published_unique_timestamps'] == i+2, 'Original graph/map counts differ')
                per_pair.append(dict(index=i, source_indices=p['source_indices'], all5_raw_byte_equal_v109=same, tail_cache_hit=bool(i),
                    prediction_seconds=p['prediction_seconds'], image_read_to_original_pose_publication_seconds=p['image_read_to_original_pose_publication_seconds'],
                    SL4_points_scale_and_LM_seconds=p['points_scale_factors_LM_seconds'], stage_seconds=call['stage_seconds'], SDK_totals=ledger))
            calls = [p['executor_call'] for p in result['pairs']]
            require(calls[0]['feature_computation_key'] == calls[1]['feature_computation_key']
                and calls[0]['input_frame_raw_sha256'][1] == calls[1]['input_frame_raw_sha256'][0]
                and calls[1]['input_frame_raw_sha256'][1] not in calls[0]['input_frame_raw_sha256'], 'Exact overlap/fresh target differs')
            z = result['executor_resources']; cache = z['previous_tail_cache']
            require(totals == dict(zip(COUNTERS,(20,20,32,42,42))) == z['SDK_totals'] and z['closed'] and not z['failed']
                and not z['active'] and not z['live_contexts'] and not z['live_output_buffers'] and not z['cleanup_errors']
                and z['calls_attempted'] == z['calls_completed'] == 2 and cache['hits'] == cache['misses'] == cache['drops'] == 1
                and cache['lookups'] == cache['commits'] == 2 and cache['cache_entries'] == cache['cache_bytes'] == 0, 'Final resources/cache not clean')
            metadata = json.loads(artifact(directory,result['map_manifest_artifact']).read_text())
            require(metadata == bm and metadata['submaps'] == 2 and metadata['graph_nodes'] == metadata['graph_factors'] == 4, 'Original map metadata differs')
            hs = array(directory,metadata['homographies'],np.float64,[4,4,4])
            require(hs.tobytes() == base_h.tobytes(), 'Original SL4 homographies differ')
            exact_files = {}
            for field in ('map_PLY_artifact','poses_artifact'):
                path = artifact(directory,result[field]); original = artifact(args.baseline_dir,base[field])
                exact_files[path.name] = sha(path) == sha(original); require(exact_files[path.name], 'Original map/poses file differs')
            poses = np.loadtxt(directory/'poses.tum')
            require(poses.shape == (3,8) and np.isfinite(poses).all() and np.allclose(poses[:,0],bm['unique_timestamps'],rtol=0,atol=1e-7), 'Three original poses differ')
            with (directory/'map.ply').open('rb') as stream:
                header = []
                while not header or header[-1] != 'end_header':
                    require(len(header)<100, 'Invalid PLY header'); header.append(stream.readline().decode('ascii').strip())
                offset = stream.tell()
            count = int(next(v.split()[-1] for v in header if v.startswith('element vertex ')))
            dtype = np.dtype([(v,'<f8') for v in ('x','y','z')]+[(v,'u1') for v in ('red','green','blue')])
            require(count == sum(sum(v['filtered_point_counts']) for v in bm['nodes']) == 609168
                and (directory/'map.ply').stat().st_size == offset+count*dtype.itemsize, 'PLY count/payload differs')
            vertices = np.memmap(directory/'map.ply',dtype=dtype,mode='r',offset=offset,shape=(count,))
            require(all(np.isfinite(vertices[v]).all() for v in ('x','y','z')), 'Map coordinates nonfinite'); del vertices
            obs = control['observations']; temps = [v for row in obs for v in row['temperatures_c'].values()]
            require(control['child_returncode'] == 0 and control['restore_errors'] == [] and control['frequency_governor_limits_restored']
                and control['restored'] == control['baseline'] and control['requested_frequency_hz'] == 1000000000
                and obs and all(v['actual_frequency_hz'] == 1000000000 for v in obs) and max(temps)<75, 'Controller/clock restoration differs')
            expected_files = {'result.json','map.json','map.ply','poses.tum','SL4-homographies.npy'} | {f'pair{i}-{k}.npy' for i in range(2) for k in SHAPES}
            require({v.name for v in directory.iterdir() if v.is_file()} == expected_files, 'Missing or unexpected saved output file')
            routes[str(mask)] = dict(requested_core_mask=mask, pairs=per_pair, all10_raw_byte_equal_v109=True,
                map_PLY_and_poses_file_byte_equal_v109=exact_files, all4_SL4_homographies_byte_equal_v109=True,
                original_graph_nodes=4, original_factors=4, submaps=2, unique_timestamps=3, PLY_vertices=count,
                SDK_totals=totals, final_cache=cache, source_and_cleanup_restoration_passed=True,
                controller_observations=len(obs), observed_peak_temperature_c=max(temps), restored_NPU_hz=int(control['restored']['cur_freq']))
        ratios = [{ 'pair_index':i, 'core7_prediction_reduction_vs_core1_percent':100*(1-routes['7']['pairs'][i]['prediction_seconds']/routes['1']['pairs'][i]['prediction_seconds'])} for i in range(2)]
        report.update(status='passed_independent_saved_tail_DINO_reuse_audit', routes=routes, verified_arrays=arrays,
            all_routes_complete_outputs_and_maps_byte_equal_v109=True, core7_vs_core1_two_matched_observation_deltas=ratios,
            prepared_pixel_SHA_independently_recomputed=False,
            limits=['Two sequential pairs per route, no repeated-sample statistical speed or paced fresh-stream/P95/20Hz claim.',
                'Prepared pixels/cache payload were not separately saved; exact input/cache identity is checked across frozen source and board receipts, while source PNGs and every saved prediction are independently rehashed.',
                'Core7 was requested and accepted with clean calls, but these host receipts do not prove physical simultaneous three-core utilization.',
                'Graph residuals and byte-equivalence show same original short-chain behavior, not trajectory accuracy, metric scale, map cadence, 30-minute or 8GB acceptance.'])
    except BaseException as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
    finally:
        report['file_sha256'] = {str(p):h for p,h in pins.items()}
        changed = [str(p) for p,h in pins.items() if not p.is_file() or sha(p) != h]
        report['saved_files_unchanged'] = not changed
        if changed: report.update(status='failed', changed_files=changed)
        os.environ.clear(); os.environ.update(oldenv)
        with args.output.open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','all_routes_complete_outputs_and_maps_byte_equal_v109')}))
    return 0 if report['status'].startswith('passed_') else 1


if __name__ == '__main__': raise SystemExit(main())
